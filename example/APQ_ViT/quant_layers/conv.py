import sys
import paddle
import numpy as np
from numpy import not_equal
from quant_layers.linear import MinMaxQuantLinear
from itertools import product
import paddle.nn.functional as F

class MinMaxQuantConv2d(paddle.nn.Conv2D):
    """
    MinMax quantize weight and output
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
        stride=1, padding=0, dilation=1, groups: int=1, bias: bool=True,
        padding_mode: str='zeros', mode='raw', w_bit=8, a_bit=8, bias_bit=None
        ):
        super().__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, padding_mode=padding_mode, bias_attr=bias)
        self.n_calibration_steps = 2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit = bias_bit
        assert bias_bit is None, 'No support bias bit now'
        self.w_interval = None
        self.a_interval = None
        self.bias_interval = None
        self.raw_input = None
        self.raw_out = None
        self.metric = None
        self.next_nodes = []
        self.w_qmax = 2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** (self.a_bit - 1)
        self.scale_initialized = None

    def forward(self, x):
        if self.mode == 'raw':
            out = paddle.nn.functional.conv2d(x=x, weight=self.weight, bias
                =self.bias, stride=self._stride, padding=self._padding,
                dilation=self._dilation, groups=self._groups)
        elif self.mode == 'quant_forward':
            out = self.quant_forward(x)
        elif self.mode == 'quant_weight_forward':
            out = self.quant_weight_forward(x)
        elif self.mode == 'calibration_step1':
            out = self.calibration_step1(x)
        elif self.mode == 'calibration_step2':
            out = self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out

    def quant_input(self, x):
        x_sim = (x / self.a_interval).round_().clip_(min=-self.a_qmax, max=
            self.a_qmax - 1)
        x_sim.multiply_(y=paddle.to_tensor(self.a_interval))
        return x_sim

    def calibration_step1(self, x):
        out = paddle.nn.functional.conv2d(x=x, weight=self.weight, bias=
            self.bias, stride=self._stride, padding=self._padding, dilation=
            self._dilation, groups=self._groups)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out


class ChannelwiseBatchingQuantConv2d(MinMaxQuantConv2d):
    """
    Only implemented acceleration with batching_calibration_step2

    setting a_bit to >= 32 will use minmax quantization, which means turning off activation quantization
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
        stride=1, padding=0, dilation=1, groups: int=1, bias: bool=True,
        padding_mode: str='zeros', mode='raw', w_bit=8, a_bit=8, bias_bit=
        None, metric='L2_norm', search_round=1, eq_alpha=0.1, eq_beta=2,
        eq_n=100, parallel_eq_n=10, n_V=1, n_H=1, init_layerwise=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=
            stride, padding=padding, dilation=dilation, groups=groups, bias
            =bias, padding_mode=padding_mode, mode=mode, w_bit=w_bit, a_bit
            =a_bit, bias_bit=bias_bit)
        self.n_V = self._out_channels
        self.n_H = 1
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.init_layerwise = init_layerwise
        self.raw_grad = None
        self.w_zerop = None
        self.a_zerop = None

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(tuple(self.raw_input.shape)[0])
        self.calib_batch_size = int(tuple(self.raw_input.shape)[0])
        while True:
            numel = 2 * (self.raw_input.size + self.raw_out.size
                ) / self.calib_size * self.calib_batch_size
            self.parallel_eq_n = int(15 * 1024 * 1024 * 1024 / 4 // numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _initialize_intervals(self):
        if self.scale_initialized != True:
            if self.init_layerwise:
                self.w_interval = (self.weight.abs().max() / (self.w_qmax -
                    0.5)).reshape([1, 1, 1, 1]).tile(repeat_times=[self.
                    out_channels, 1, 1, 1])
            else:
                self.w_interval = self.weight.abs().amax(axis=[1, 2, 3],
                    keepdim=True) / (self.w_qmax - 0.5)
            tmp_a_intervals = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
                a_interval_ = (x_.abs().max() / (self.a_qmax - 0.5)).detach(
                    ).reshape([1, 1])
                tmp_a_intervals.append(a_interval_)
            self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1).amax(
                axis=1, keepdim=False)
        self.scale_initialized = True

    def _initialize_a_interval(self):
        tmp_a_intervals = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
            a_interval_ = (x_.abs().max() / (self.a_qmax - 0.5)).detach().reshape(
                [1, 1])
            tmp_a_intervals.append(a_interval_)
        self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1).amax(axis
            =1, keepdim=False)

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *, features
        """
        if metric == "info":
            _reshape = False
            if len(tensor_raw.shape) == 5:
                _reshape = True
                tensor_raw = tensor_raw.reshape([tensor_raw.shape[0], tensor_raw.shape[1], tensor_raw.shape[2], -1])
                tensor_sim = tensor_sim.reshape([tensor_sim.shape[0], tensor_sim.shape[1], tensor_sim.shape[2], -1])
            feat = tensor_raw - tensor_sim
            BINS = 200
            feat_min = paddle.min(feat, axis=-1, keepdim=True)
            feat_max = paddle.max(feat, axis=-1, keepdim=True)
            delta_E = (feat_max - feat_min) / float(BINS-1)
            index = paddle.floor((feat - feat_min) / delta_E)
            index = index.transpose([1, 0])  # transpose(0, 1) in Paddle
            index = index.reshape([index.shape[0], -1])
           
            similarity = paddle.zeros([index.shape[0]])
            idx = 0 
            for e in index:
                e = paddle.unique(e, return_counts=True)[1]
                p = e / paddle.sum(e)
                similarity[idx] = paddle.distribution.Categorical(probs=p).entropy()
                idx += 1
            scale_index = paddle.norm(feat - paddle.mean(feat, axis=-1, keepdim=True), p=1, axis=-1)
            similarity = -similarity.reshape([1, similarity.shape[0], 1]) * scale_index
            if _reshape:
                similarity = similarity.unsqueeze([3, 4])
        elif metric == "cosine":
            # support cosine on patch dim, which is sub-optimal
            # not supporting search best a interval
            b, parallel_eq_n, oc = tensor_sim.shape[0], tensor_sim.shape[1], tensor_sim.shape[2]
            similarity = F.cosine_similarity(tensor_raw.reshape([b, 1, oc, -1]), tensor_sim.reshape([b, parallel_eq_n, oc, -1]), axis=-1).reshape([b, parallel_eq_n, oc, 1, 1])
        else:
            if metric == "L1_norm":
                similarity = -paddle.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -paddle.abs(tensor_raw) * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                assert raw_grad is not None, f"raw_grad is None in _get_similarity!"
                raw_grad = paddle.reshape(raw_grad, shape=tensor_raw.shape)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                return self._get_similarity(tensor_raw=tensor_raw, tensor_sim=tensor_sim, metric="cosine", raw_grad=raw_grad)
                # raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity

    def _search_best_w_interval(self, weight_interval_candidates=None):
        batch_similarities = []
        w_max = self.weight.amax(axis=[1, 2, 3], keepdim=True)
        w_min = self.weight.amin(axis=[1, 2, 3], keepdim=True)
        new_max = paddle.to_tensor(data=[(self.eq_alpha + i * (self.eq_beta -
            self.eq_alpha) / self.eq_n) for i in range(self.eq_n + 1)]).cuda(
            blocking=True).reshape([-1, 1, 1, 1, 1]) * w_max
        new_min = paddle.to_tensor(data=[(self.eq_alpha + i * (self.eq_beta -
            self.eq_alpha) / self.eq_n) for i in range(self.eq_n + 1)]).cuda(
            blocking=True).reshape([-1, 1, 1, 1, 1]) * w_min
        new_scale = (new_max - new_min) / float(self.w_qmax * 2 - 1)
        new_scale.clip_(min=1e-08)
        new_zero_point = -self.w_qmax - paddle.round(new_min / new_scale)
        new_zero_point.clip_(min=-self.w_qmax, max=self.w_qmax - 1)
        w_zeropoint_candidates = new_zero_point.clone().detach().cuda(blocking
            =True)
        weight_interval_candidates = new_scale.clone().detach().cuda(blocking
            =True)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda(blocking=True)
            raw_out = self.raw_out[b_st:b_ed].cuda(blocking=True).unsqueeze(
                axis=1)
            raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True
                ) if self.raw_grad is not None else None
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_w_interval = weight_interval_candidates[p_st:p_ed]
                cur_w_zero = w_zeropoint_candidates[p_st:p_ed]
                oc, ic, kw, kh = tuple(self.weight.data.shape)
                w_sim = self.weight.unsqueeze(axis=0)
                w_int = paddle.round(w_sim / cur_w_interval + cur_w_zero)
                w_int = paddle.clip(x=w_int, min=-self.w_qmax, max=self.
                    w_qmax - 1)
                w_sim = (w_int - cur_w_zero) * cur_w_interval
                del w_int, cur_w_interval, cur_w_zero
                w_sim = w_sim.reshape([-1, ic, kw, kh])
                bias_sim = self.bias.tile(repeat_times=p_ed - p_st
                    ) if self.bias is not None else None
                x_sim = self.quant_input(x) if self.a_bit < 32 else x
                out_sim = paddle.nn.functional.conv2d(x=x_sim, weight=w_sim,
                    bias=bias_sim, stride=self._stride, padding=self._padding,
                    dilation=self._dilation, groups=self._groups)
                out_sim = paddle.concat(x=paddle.chunk(x=out_sim.unsqueeze(
                    axis=1), chunks=p_ed - p_st, axis=2), axis=1)
                similarity = self._get_similarity(raw_out, out_sim, self.
                    metric, raw_grad)
                similarity = paddle.mean(x=similarity, axis=[3, 4])
                similarity = paddle.sum(x=similarity, axis=0, keepdim=True)
                similarities.append(similarity)
            similarities = paddle.concat(x=similarities, axis=1)
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(x=batch_similarities, axis=0).sum(
            axis=0, keepdim=False)
        best_index = batch_similarities.argmax(axis=0).reshape([1, -1, 1, 1, 1])
        self.w_interval = paddle.take_along_axis(arr=
            weight_interval_candidates, axis=0, indices=best_index,
            broadcast=False).squeeze(axis=0)
        self.w_zerop = paddle.take_along_axis(arr=w_zeropoint_candidates,
            axis=0, indices=best_index, broadcast=False).squeeze(axis=0)

    def _search_best_a_interval(self, input_interval_candidates):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]  # Paddle 默认在GPU上

            x_max = (paddle.max(x)).detach().reshape([1, 1])
            x_min = (paddle.min(x)).detach().reshape([1, 1])
            new_max = paddle.to_tensor(
                [self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n for i in range(self.eq_n + 1)],
                dtype=x.dtype
            ).reshape([-1, 1, 1, 1, 1]) * x_max
            new_min = paddle.to_tensor(
                [self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n for i in range(self.eq_n + 1)],
                dtype=x.dtype
            ).reshape([-1, 1, 1, 1, 1]) * x_min

            new_scale = (new_max - new_min) / float(self.a_qmax * 2 - 1)
            new_scale = paddle.clip(new_scale, min=1e-8)
            new_zero_point = -self.a_qmax - paddle.round(new_min / new_scale)
            new_zero_point = paddle.clip(new_zero_point, min=-self.a_qmax, max=self.a_qmax - 1)
            a_zeropoint_candidates = new_zero_point
            input_interval_candidates = new_scale

            raw_out = self.raw_out[b_st:b_ed].unsqueeze(1)  # shape: b, 1, oc, fw, fh
            raw_grad = self.raw_grad[b_st:b_ed]
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # shape: parallel_eq_n, 1, 1, 1, 1
                cur_a_interval = input_interval_candidates[p_st:p_ed]
                cur_a_zero = a_zeropoint_candidates[p_st:p_ed]

                # quantize weight and bias
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                B, ic, iw, ih = x.shape
                x_sim = x.unsqueeze(0)  # shape: 1, b, ic, iw, ih

                # asym uniform affine
                x_int = paddle.round(x_sim / cur_a_interval)
                x_int = paddle.clip(x_int + cur_a_zero, min=-self.a_qmax, max=self.a_qmax - 1)
                x_sim = (x_int - cur_a_zero) * cur_a_interval
                del x_int, cur_a_interval, cur_a_zero

                # shape: parallel_eq_n*b, ic, iw, ih
                x_sim = x_sim.reshape([-1, ic, iw, ih])
                # calculate similarity and store them
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)  # shape: parallel_eq_n*b, oc, fw, fh
                out_sim = paddle.concat(paddle.chunk(out_sim.unsqueeze(0), chunks=p_ed - p_st, dim=1), dim=0)  # shape: parallel_eq_n, b, oc, fw, fh
                out_sim = out_sim.transpose([0, 1])  # shape: b, parallel_eq_n, oc, fw, fh
                # shape: b, parallel_eq_n, oc, fw, fh
                similarity = self._get_similarity(raw_out, out_sim, metric, raw_grad=raw_grad)
                # shape: b, parallel_eq_n
                similarity = paddle.mean(similarity, axis=[2, 3, 4])
                # shape: 1, parallel_eq_n
                similarity = paddle.sum(similarity, axis=0, keepdim=True)
                similarities.append(similarity)

            similarities = paddle.concat(similarities, axis=1)  # shape: 1, eq_n
            batch_similarities.append(similarities)

        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False)  # shape: eq_n
        a_best_index = paddle.argmax(batch_similarities, axis=0).reshape([1, 1, 1, 1, 1])
        self.a_interval = paddle.gather(input_interval_candidates, axis=0, index=a_best_index).squeeze()
        self.a_zerop = paddle.gather(a_zeropoint_candidates, axis=0, index=a_best_index).squeeze()
    
    def calibration_step2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        weight_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1]
            ) * self.w_interval.unsqueeze(axis=0)
        input_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1]
            ) * self.a_interval
        for e in range(self.search_round):
            self._search_best_w_interval(weight_interval_candidates)
            if self.a_bit < 32:
                self._search_best_a_interval(input_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad

    def calibration_step2_weight(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        weight_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1]
            ) * self.w_interval.unsqueeze(axis=0)
        for e in range(self.search_round):
            self._search_best_w_interval_alternative(weight_interval_candidates
                )
        self.calibrated = True
        self.raw_input, self.raw_out, self.raw_grad = None, None, None

    def calibration_step2_weight2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        weight_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1]
            ) * self.w_interval.unsqueeze(axis=0)
        for e in range(self.search_round):
            self._search_best_w_interval(weight_interval_candidates)
        self.calibrated = True
        self.raw_input, self.raw_out, self.raw_grad = None, None, None

    def calibration_step2_act(self):
        self._initialize_calib_parameters()
        self._initialize_a_interval()
        input_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1]
            ) * self.a_interval
        for e in range(self.search_round):
            if self.a_bit < 32:
                self._search_best_a_interval(input_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad

    def quant_weight_bias(self):
        w_sim = (self.weight / self.w_interval).round_().clip(min=-self.
            w_qmax, max=self.w_qmax - 1).multiply_(y=paddle.to_tensor(self.
            w_interval))
        return w_sim, self.bias

    def quant_forward(self, x):
        assert self.calibrated is not None, f'You should run calibrate_forward before run quant_forward for {self}'
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x) if self.a_bit < 32 else x
        out = paddle.nn.functional.conv2d(x=x_sim, weight=w_sim, bias=
            bias_sim, stride=self._stride, padding=self._padding, dilation=
            self._dilation, groups=self._groups)
        return out

    def quant_weight_forward(self, x):
        assert self.calibrated is not None, f'You should run calibrate_forward before run quant_forward for {self}'
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = x
        out = paddle.nn.functional.conv2d(x=x_sim, weight=w_sim, bias=
            bias_sim, stride=self.stride, padding=self.padding, dilation=
            self.dilation, groups=self.groups)
        return out

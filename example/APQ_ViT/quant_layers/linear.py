import sys
import paddle
import paddle.nn.functional as F
from quant_layers.matmul import PTQSLBatchingQuantMatMul
import numpy as np
import logging
import sys
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=
    log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


class MinMaxQuantLinear(paddle.nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True,
        mode='raw', w_bit=8, a_bit=8, bias_bit=None, bias_correction=False):
        super().__init__(in_features, out_features, bias)
        self.n_calibration_step = 2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit = bias_bit
        assert bias_bit is None, 'No support bias bit now'
        self.w_interval = None
        self.a_interval = None
        self.raw_input = None
        self.raw_out = None
        self.metric = None
        self.next_nodes = []
        self.w_qmax = 2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** (self.a_bit - 1)
        self.bias_correction = bias_correction
        self.scale_initialized = None
        self.w_zerop = None
        self.a_zerop = None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        if self.mode == 'raw':
            out = paddle.nn.functional.linear(x=x, weight=self.weight,
                bias=self.bias)
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

    def quant_forward(self, x):
        assert self.calibrated is not None, f'You should run calibrate_forward before run quant_forward for {self}'
        w_sim, bias_sim = self.quant_weight_bias()
        # x_sim = self.quant_input(x)
        x_sim = x
        out = paddle.nn.functional.linear(x=x_sim, weight=w_sim.T, bias=bias_sim)
        return out

    def quant_weight_forward(self, x):
        assert self.calibrated is not None, f'You should run calibrate_forward before run quant_forward for {self}'
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = x
        out = paddle.nn.functional.linear(x=x_sim, weight=w_sim.T, bias=
            bias_sim)
        return out

    def _bias_correction_quant_forward(self, x):
        if self.bias_correction and self.bias != None:
            w_sim = self.quant_weight_bias()[0]
            x_sim = self.quant_input(x)
            eps = paddle.nn.functional.linear(x=x_sim, weight=(w_sim - self
                .weight.data).T, bias=None)
            eps = paddle.mean(x=eps, axis=list(range(len(tuple(eps.shape)) -
                1)), keepdim=False)
            self.bias -= eps
            self.bias_correction = False
        return self.quant_forward(x)

    def calibration_step1(self, x):
        out = paddle.nn.functional.linear(x=x, weight=self.weight.T, bias=
            self.bias)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out

    def quant_weight_bias(self):
        if self.w_zerop is not None:
            w = ((self.weight.reshape([self.n_V, self.crb_rows, self.n_H, self.
                crb_cols]) / self.w_interval).round_() + self.w_zerop).clip_(min
                =-self.w_qmax, max=self.w_qmax - 1)
            w_sim = (w - self.w_zerop).multiply_(y=paddle.to_tensor(self.
                w_interval)).reshape([self.out_features, self.in_features])
        else:
            w = (self.weight.reshape([self.n_V, self.crb_rows, self.n_H, self.
                crb_cols]) / self.w_interval).round_().clip_(min=-self.
                w_qmax, max=self.w_qmax - 1)
            w_sim = w.multiply_(y=paddle.to_tensor(self.w_interval)).reshape([self
                .out_features, self.in_features])
        if self.bias is not None:
            return w_sim, self.bias
        else:
            return w_sim, None

    def quant_input(self, x):
        return x
        x_sim = paddle.concat(x=paddle.chunk(x=x.unsqueeze(axis=-2), chunks
            =self.n_a, axis=-1), axis=-2)
        if self.a_zerop is not None:
            x_sim = (x_sim.divide_(y=paddle.to_tensor(self.a_interval)).
                round_() + self.a_zerop).clip_(min=-self.a_qmax, max=self.
                a_qmax - 1)
            x_sim = (x_sim - self.a_zerop).multiply_(y=paddle.to_tensor(
                self.a_interval)).reshape(x.shape)
        else:
            x_sim = x_sim.divide_(y=paddle.to_tensor(self.a_interval)).round_(
                ).clip_(min=-self.a_qmax, max=self.a_qmax - 1)
            x_sim = x_sim.multiply_(y=paddle.to_tensor(self.a_interval)
                ).reshape(x.shape)
        return x_sim


class PTQSLBatchingQuantLinear(MinMaxQuantLinear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True,
        mode='raw', w_bit=8, a_bit=8, bias_bit=None, bias_correction=False,
        metric='L2_norm', search_round=1, eq_alpha=0, eq_beta=1, eq_n=100,
        parallel_eq_n=10, n_H=1, n_V=1, n_a=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode,
            w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=
            bias_correction)
        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = False
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.n_H = n_H
        self.n_V = n_V
        self.n_a = n_a
        self.crb_rows = out_features // n_V
        self.crb_cols = in_features // n_H
        self.crb_acts = in_features // n_a
        self.parallel_eq_n = parallel_eq_n
        self.init_layerwise = init_layerwise
        self.raw_grad = None
        self.a_zerop = None
        self.w_zerop = None

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(tuple(self.raw_input.shape)[0])
        self.calib_batch_size = int(tuple(self.raw_input.shape)[0])
        while True:
            numel = 2 * (self.raw_input.size + self.raw_out.size
                ) / self.calib_size * self.calib_batch_size
            self.parallel_eq_n = int(3 * 1024 * 1024 * 1024 / 4 // numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _initialize_intervals(self):
        if self.scale_initialized != True:
            if self.init_layerwise:
                self.w_interval = (self.weight.abs().max() / (self.w_qmax -
                    0.5)).view(1, 1, 1, 1).tile(repeat_times=[self.n_V, 1,
                    self.n_H, 1])
            else:
                self.w_interval = self.weight.reshape([self.n_V, self.crb_rows,
                    self.n_H, self.crb_cols]).abs().amax(axis=[1, 3],
                    keepdim=True) / (self.w_qmax - 0.5)
            tmp_a_intervals = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
                if self.init_layerwise:
                    a_interval_ = (x_.abs().max() / (self.a_qmax - 0.5)
                        ).detach().view(1, 1).tile(repeat_times=[self.n_a, 1])
                else:
                    axes = list(range(len(x_.shape) - 1)) + [-1]
                    a_interval_ = (
                        x_.reshape([*tuple(x_.shape)[:-1], self.n_a, self.crb_acts])
                        .abs()
                        .amax(axis=axes, keepdim=False)
                        / (self.a_qmax - 0.5)
                    ).unsqueeze(axis=-1)
                    # a_interval_ = (x_.view(*tuple(x_.shape)[:-1], self.n_a,
                    #     self.crb_acts).abs().amax(axis=list(range(len(tuple
                    #     (x_.shape)) - 1)) + [-1], keepdim=False) / (self.
                    #     a_qmax - 0.5)).unsqueeze(axis=-1)
                tmp_a_intervals.append(a_interval_)
            self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1).amax(
                axis=1, keepdim=True)
        self.scale_initialized = True

    def _initialize_a_interval(self):
        tmp_a_intervals = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
            if self.init_layerwise:
                a_interval_ = (x_.abs().max() / (self.a_qmax - 0.5)).detach(
                    ).view(1, 1).tile(repeat_times=[self.n_a, 1])
            else:
                a_interval_ = (x_.view(*tuple(x_.shape)[:-1], self.n_a,
                    self.crb_acts).abs().amax(axis=list(range(len(tuple(x_.
                    shape)) - 1)) + [-1], keepdim=False) / (self.a_qmax - 0.5)
                    ).unsqueeze(axis=-1)
            tmp_a_intervals.append(a_interval_)
        self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1).amax(axis
            =1, keepdim=True)



    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "r2b":
            Q_s_norm = tensor_raw / paddle.norm(tensor_raw, p=2)
            Q_t_norm = tensor_sim / paddle.norm(tensor_sim, p=2)
            tmp = Q_s_norm - Q_t_norm
            similarity = -paddle.norm(tmp, p=2, axis=-1, keepdim=True)
        elif metric == "info":
            feat = tensor_raw - tensor_sim
            if len(feat.shape) == 5:
                feat = feat.transpose([1, 2]).transpose([2, 3])
                similarity = paddle.zeros(feat.shape[:3])
                a1, a2, a3 = feat.shape[:3]
                feat = feat.reshape([a1, a2, a3, -1])
                thresh = paddle.std(feat, axis=-1) * 3
                for b1 in range(a1):
                    for b2 in range(a2):
                        for b3 in range(a3):
                            outliers = paddle.abs(feat[b1][b2][b3])
                            outliers[outliers < thresh[b1][b2][b3]] = 0
                            similarity[b1][b2][b3] = -paddle.sum(outliers) * thresh[b1][b2][b3]
            elif len(feat.shape) == 4:
                feat = feat.transpose([1, 2])
                similarity = paddle.zeros(feat.shape[:2])
                a1, a2 = feat.shape[:2]
                feat = feat.reshape([a1, a2, -1])
                thresh = paddle.std(feat, axis=-1) * 3
                for b1 in range(a1):
                    for b2 in range(a2):
                        outliers = paddle.abs(feat[b1][b2])
                        outliers[outliers < thresh[b1][b2]] = 0
                        similarity[b1][b2] = -paddle.sum(outliers) * thresh[b1][b2]
            elif len(feat.shape) == 3:
                similarity = paddle.zeros(feat.shape[:2])
                a1, a2 = feat.shape[:2]
                thresh = paddle.std(feat, axis=-1) * 3
                for b1 in range(a1):
                    for b2 in range(a2):
                        outliers = paddle.abs(feat[b1][b2])
                        outliers[outliers < thresh[b1][b2]] = 0
                        similarity[b1][b2] = -paddle.sum(outliers) * thresh[b1][b2]
        elif metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, axis=-1)  # [1, 197, 100]
        else:
            if metric == "L1_norm":
                similarity = -paddle.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -paddle.abs(tensor_raw) * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "apq_hessian":
                assert raw_grad is not None, "raw_grad is None in _get_similarity!"
                raw_grad = paddle.reshape(raw_grad, tensor_raw.shape)

                feat = tensor_raw - tensor_sim
                diff = paddle.abs(feat)
                perc = np.percentile(diff.detach().cpu().numpy(), 10)
                feat[diff < perc] = 0
                similarity = -(raw_grad * feat) ** 2
                # raw_grad = paddle.reshape(raw_grad, shape=tensor_raw.shape)
                # similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = paddle.mean(similarity, axis=-1)
        return similarity


    def _get_pearson_w(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,n_V,crb_rows
        tensor_raw: b,*,1,n_V,crb_rows
        """
        b, parallel_eq_n, n_V = tensor_sim.shape[0], tensor_sim.shape[-3], tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose([0, 2, 3, 1]).reshape([b, -1, n_V, parallel_eq_n])
        tensor_raw = tensor_raw.transpose([0, 2, 3, 1]).reshape([b, -1, n_V, 1])
        tensor_sim_mean = paddle.mean(tensor_sim, axis=[0, 1], keepdim=True)
        tensor_raw_mean = paddle.mean(tensor_raw, axis=[0, 1], keepdim=True)
        similarity = F.cosine_similarity(tensor_raw - tensor_raw_mean, tensor_sim - tensor_sim_mean, axis=1)  # shape: b, n_V, parallel_eq_n
        similarity = similarity.transpose([0, 2, 1]).reshape([b, parallel_eq_n, n_V])
        return similarity


    def _get_pearson_a(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,oc
        tensor_raw: b,*,1,oc
        """
        b, parallel_eq_n = tensor_sim.shape[0], tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose([0, 2, 1]).reshape([b, -1, parallel_eq_n])
        tensor_raw = tensor_raw.transpose([0, 2, 1]).reshape([b, -1, 1])
        tensor_sim_mean = paddle.mean(tensor_sim, axis=[0, 1], keepdim=True)
        tensor_raw_mean = paddle.mean(tensor_raw, axis=[0, 1], keepdim=True)
        similarity = F.cosine_similarity(tensor_raw - tensor_raw_mean, tensor_sim - tensor_sim_mean, axis=1)  # shape: b, parallel_eq_n
        return similarity


    def _search_best_w_interval(self, weight_interval_candidates=None):
        tmp_w_interval = self.w_interval.unsqueeze(axis=0)
        if self.w_zerop is None:
            tmp_w_zerop = paddle.zeros_like(x=tmp_w_interval)
        else:
            tmp_w_zerop = self.w_zerop.unsqueeze(axis=0)
        w_max = self.weight.reshape([self.n_V, self.crb_rows, self.n_H, self.
            crb_cols]).amax(axis=[1, 3], keepdim=True)
        w_min = self.weight.reshape([self.n_V, self.crb_rows, self.n_H, self.
            crb_cols]).amin(axis=[1, 3], keepdim=True)
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
        w_zeropoint_candidates = new_zero_point.clone().detach().cuda(blocking=True)
        weight_interval_candidates = new_scale.clone().detach().cuda(blocking=True)
        
        for h in range(self.n_H):
            batch_similarities = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x = self.raw_input[b_st:b_ed].cuda(blocking=True)
                raw_out_expanded = self.raw_out[b_st:b_ed].cuda(blocking=True
                    ).unsqueeze(axis=-2)
                raw_out_expanded = paddle.concat(x=paddle.chunk(x=raw_out_expanded.unsqueeze(axis=-2), chunks=self.n_V, axis=-1), axis=-2)
                # print(self.raw_grad.shape)
                raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True)
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                    cur_w_interval = tmp_w_interval.tile(repeat_times=[p_ed - p_st, 1, 1, 1, 1])
                    cur_w_interval[:, :, :, h:h + 1, : ] = weight_interval_candidates[p_st:p_ed, :, :, h:h + 1, :]
                    cur_w_zero = tmp_w_zerop.tile(repeat_times=[p_ed - p_st, 1, 1, 1, 1])
                    cur_w_zero[:, :, :, h:h + 1, :] = w_zeropoint_candidates[ p_st:p_ed, :, :, h:h + 1, :]
                    w_sim = self.weight.reshape([self.n_V, self.crb_rows, self.n_H, self.crb_cols]).unsqueeze(axis=0)
                    w_int = paddle.round(w_sim / cur_w_interval + cur_w_zero)
                    w_int = paddle.clip(x=w_int, min=-self.w_qmax, max=self.w_qmax - 1)
                    w_sim = (w_int - cur_w_zero) * cur_w_interval
                    del w_int, cur_w_interval, cur_w_zero
                    w_sim = w_sim.reshape([-1, self.in_features])
                    bias_sim = self.bias.tile(repeat_times=p_ed - p_st) if self.bias is not None else None
                    x_sim = self.quant_input(x)
                    out_sim = paddle.nn.functional.linear(x=x_sim, weight=w_sim.T, bias=bias_sim)
                    out_sim = paddle.concat(x=paddle.chunk(x=out_sim.unsqueeze(axis=-2), chunks=p_ed - p_st, axis=-1), axis=-2)
                    out_sim = paddle.concat(x=paddle.chunk(x=out_sim.unsqueeze(axis=-2), chunks=self.n_V, axis=-1), axis=-2)
                    if self.metric != 'pearson':
                        # print(raw_out_expanded.shape, raw_grad.shape)
                        similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad)
                        if len(tuple(similarity.shape)) > 3:
                            similarity = paddle.mean(x=similarity, axis=list(range(1, len(similarity.shape) - 2)))
                    else:
                        similarity = self._get_pearson_w(raw_out_expanded, out_sim)
                    similarity = similarity.sum(axis=0, keepdim=True)
                    similarities.append(similarity)
                similarities = paddle.concat(x=similarities, axis=1)
                batch_similarities.append(similarities)
            batch_similarities = paddle.concat(x=batch_similarities, axis=0
                ).sum(axis=0, keepdim=False)
            h_best_index = batch_similarities.argmax(axis=0).reshape([1, -1,
                1, 1, 1]).cuda(blocking=True)
            tmp_w_interval[:, :, :, h:h + 1, :] = paddle.take_along_axis(arr
                =weight_interval_candidates[:, :, :, h:h + 1, :], axis=0,
                indices=h_best_index, broadcast=False)
            tmp_w_zerop[:, :, :, h:h + 1, :] = paddle.take_along_axis(arr=
                w_zeropoint_candidates[:, :, :, h:h + 1, :], axis=0,
                indices=h_best_index, broadcast=False)
        # import pdb; pdb.set_trace()
        self.w_interval = tmp_w_interval.squeeze(axis=0)
        self.w_zerop = tmp_w_zerop.squeeze(axis=0)

        # print(f"w interval: {self.w_interval}")
        # print(f"w zeropoint: {self.w_zerop}")

    def _search_best_a_interval(self, input_interval_candidates=None):
        tmp_a_interval = self.a_interval.unsqueeze(axis=-1)
        if self.a_zerop is None:
            tmp_a_zerop = paddle.zeros_like(x=tmp_a_interval)
        else:
            tmp_a_zerop = self.a_zerop.unsqueeze(axis=-1)
        for a in range(self.n_a):
            batch_similarities = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x = self.raw_input[b_st:b_ed].cuda(blocking=True)
                x_max = x.reshape([*tuple(x.shape)[:-1], self.n_a, self.crb_acts]
                    ).amax(axis=list(range(len(tuple(x.shape)) - 1)) + [-1],
                    keepdim=False).unsqueeze(axis=-1)
                x_min = x.reshape([*tuple(x.shape)[:-1], self.n_a, self.crb_acts]
                    ).amin(axis=list(range(len(tuple(x.shape)) - 1)) + [-1],
                    keepdim=False).unsqueeze(axis=-1)
                new_max = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([1, 1, -1]) * x_max
                new_min = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([1, 1, -1]) * x_min
                new_scale = (new_max - new_min) / float(self.a_qmax * 2 - 1)
                new_scale.clip_(min=1e-08)
                new_zero_point = -self.a_qmax - paddle.round(new_min /
                    new_scale)
                new_zero_point.clip_(min=-self.a_qmax, max=self.a_qmax - 1)
                a_zeropoint_candidates = new_zero_point.cuda(blocking=True)
                input_interval_candidates = new_scale.cuda(blocking=True)
                raw_out_expanded = self.raw_out[b_st:b_ed].cuda(blocking=True
                    ).unsqueeze(axis=-2)
                raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True)
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                    cur_a_interval = tmp_a_interval.tile(repeat_times=[1, 1,
                        p_ed - p_st])
                    cur_a_interval[a:a + 1, :, :] = input_interval_candidates[a
                        :a + 1, :, p_st:p_ed]
                    cur_a_zero = tmp_a_zerop.tile(repeat_times=[1, 1, p_ed -
                        p_st])
                    cur_a_zero[a:a + 1, :, :] = a_zeropoint_candidates[a:a +
                        1, :, p_st:p_ed]
                    w_sim, bias_sim = self.quant_weight_bias()
                    x_sim = paddle.concat(x=paddle.chunk(x=x.unsqueeze(axis
                        =-2), chunks=self.n_a, axis=-1), axis=-2).unsqueeze(
                        axis=-1)
                    x_int = paddle.round(x_sim / cur_a_interval)
                    x_int = paddle.clip(x=x_int + cur_a_zero, min=-self.
                        a_qmax, max=self.a_qmax - 1)
                    x_sim = (x_int - cur_a_zero) * cur_a_interval
                    del x_int, cur_a_interval, cur_a_zero
                    x_sim = x_sim.transpose(perm=[*list(range(len(tuple(
                        x_sim.shape)) - 3)), -1, -3, -2]).reshape([*tuple(x.
                        shape)[:-1], p_ed - p_st, tuple(x.shape)[-1]])
                    out_sim = paddle.nn.functional.linear(x=x_sim, weight=
                        w_sim.T, bias=bias_sim)
                    if self.metric != 'pearson':
                        similarity = self._get_similarity(raw_out_expanded,
                            out_sim, self.metric, raw_grad)
                        if len(tuple(similarity.shape)) > 2:
                            similarity = paddle.mean(x=similarity, axis=
                                list(range(1, len(tuple(similarity.shape)) -
                                1)))
                    else:
                        similarity = self._get_pearson_a(raw_out_expanded,
                            out_sim)
                    similarity = paddle.sum(x=similarity, axis=0, keepdim=True)
                    similarities.append(similarity)
                similarities = paddle.concat(x=similarities, axis=1)
                batch_similarities.append(similarities.cuda(blocking=True))
            batch_similarities = paddle.concat(x=batch_similarities, axis=0
                ).sum(axis=0, keepdim=False)
            a_best_index = batch_similarities.argmax(axis=0, keepdim=True
                ).reshape([1, 1, -1]).cuda(blocking=True)
            tmp_a_interval[a:a + 1, :, :] = paddle.take_along_axis(arr=
                input_interval_candidates[a:a + 1, :, :], axis=2, indices=
                a_best_index, broadcast=False)
            tmp_a_zerop[a:a + 1, :, :] = paddle.take_along_axis(arr=
                a_zeropoint_candidates[a:a + 1, :, :], axis=2, indices=
                a_best_index, broadcast=False)
        # import pdb; pdb.set_trace()
        self.a_interval = tmp_a_interval.squeeze(axis=-1)
        self.a_zerop = tmp_a_zerop.squeeze(axis=-1)
        
        # print(f"a interval: {self.a_interval}")
        # print(f"a zeropoint: {self.a_zerop}")

    def calibration_step2(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()

        weight_interval_candidates = paddle.to_tensor(
            [(self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(self.eq_n + 1)])
        weight_interval_candidates = weight_interval_candidates.reshape([-1, 1, 1, 1, 1])
        weight_interval_candidates = weight_interval_candidates * self.w_interval.unsqueeze(axis=0)

        # weight_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
        #     i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
        #     self.eq_n + 1)]).cuda(blocking=True).view(-1, 1, 1, 1, 1
        #     ) * self.w_interval.unsqueeze(axis=0)
        
        input_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).reshape([1, 1, -1]
            ) * self.a_interval.unsqueeze(axis=-1)
        for e in range(self.search_round):
            self._search_best_w_interval(weight_interval_candidates)
            self._search_best_a_interval(input_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        return None

    def calibration_step2_weight(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()
        for e in range(self.search_round):
            self._search_best_w_interval_alternative()
        self.calibrated = True
        self.raw_input, self.raw_out, self.raw_grad = None, None, None
        return None

    def calibration_step2_weight2(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()
        for e in range(self.search_round):
            self._search_best_w_interval()
        self.calibrated = True
        self.raw_input, self.raw_out, self.raw_grad = None, None, None
        return None

    def calibration_step2_act(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_a_interval()
        input_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha +
            i * (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(
            self.eq_n + 1)]).cuda(blocking=True).view(1, 1, -1
            ) * self.a_interval.unsqueeze(axis=-1)
        for e in range(self.search_round):
            self._search_best_a_interval(input_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        return None


class PostGeluPTQSLBatchingQuantLinear(PTQSLBatchingQuantLinear):
    """ 
    An Agile implementation of PostGeluPTQSLBatchingQuantLinear
    use a_interval for positive activation quantization and a_neg_interval for negative activation quantization
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True,
        mode='raw', w_bit=8, a_bit=8, bias_bit=None, bias_correction=False,
        metric='L2_norm', search_round=1, eq_alpha=0, eq_beta=1, eq_n=100,
        parallel_eq_n=10, n_H=1, n_V=1, n_a=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode,
            w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=
            bias_correction, metric=metric, search_round=search_round,
            eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=
            parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a, init_layerwise=
            init_layerwise)
        self.a_neg_interval = 0.16997124254703522 / self.a_qmax
        self.log_n = 0.1996

    def _initialize_intervals(self):
        if self.scale_initialized != True:
            if self.init_layerwise:
                self.w_interval = (self.weight.abs().max() / (self.w_qmax -
                    0.5)).view(1, 1, 1, 1).tile(repeat_times=[self.n_V, 1,
                    self.n_H, 1])
            else:
                self.w_interval = paddle.abs(paddle.reshape(self.weight, [self.n_V, self.crb_rows, self.n_H, self.crb_cols])).amax(axis=[1, 3], keepdim=True) / (self.w_qmax - 0.5)
                # self.w_interval = self.weight.view(self.n_V, self.crb_rows,
                #     self.n_H, self.crb_cols).abs().amax(axis=[1, 3],
                #     keepdim=True) / (self.w_qmax - 0.5)
            if self.init_layerwise:
                tmp_a_intervals = []
                for b_st in range(0, self.calib_size, self.calib_batch_size):
                    b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                    x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
                    a_interval_ = (x_.max() / (self.a_qmax - 0.5)).detach(
                        ).view(1, 1).tile(repeat_times=[self.n_a, 1])
                    tmp_a_intervals.append(a_interval_)
                self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1
                    ).amax(axis=1, keepdim=True)
            else:
                tmp_a_intervals = []
                for b_st in range(0, self.calib_size, self.calib_batch_size):
                    b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                    x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
                    # a_interval_ = (x_.view(*tuple(x_.shape)[:-1], self.n_a,
                    #     self.crb_acts).amax(axis=list(range(len(tuple(x_.
                    #     shape)) - 1)) + [-1], keepdim=False) / (self.a_qmax -
                    #     0.5)).unsqueeze(axis=-1)
                    viewed_x = x_.view(tuple(x_.shape)[:-1] + (self.n_a, self.crb_acts))
                    amax_result = paddle.amax(viewed_x, axis=tuple(range(len(tuple(x_.shape)) - 1)) + (-1,), keepdim=False)
                    scaled_result = amax_result / (self.a_qmax - 0.5)
                    a_interval_ = paddle.unsqueeze(scaled_result, axis=-1)

                    tmp_a_intervals.append(a_interval_)
                self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1
                    ).amax(axis=1, keepdim=True)
        self.scale_initialized = True

    def _initialize_a_interval(self):
        if self.init_layerwise:
            tmp_a_intervals = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
                a_interval_ = (x_.max() / (self.a_qmax - 0.5)).detach().view(
                    1, 1).tile(repeat_times=[self.n_a, 1])
                tmp_a_intervals.append(a_interval_)
            self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1).amax(
                axis=1, keepdim=True)
        else:
            tmp_a_intervals = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x_ = self.raw_input[b_st:b_ed].cuda(blocking=True)
                a_interval_ = (x_.view(*tuple(x_.shape)[:-1], self.n_a,
                    self.crb_acts).amax(axis=list(range(len(tuple(x_.shape)
                    ) - 1)) + [-1], keepdim=False) / (self.a_qmax - 0.5)
                    ).unsqueeze(axis=-1)
                tmp_a_intervals.append(a_interval_)
            self.a_interval = paddle.concat(x=tmp_a_intervals, axis=1).amax(
                axis=1, keepdim=True)

    def quant_input(self, x):
        x_ = paddle.concat(x=paddle.chunk(x=x.unsqueeze(axis=-2), chunks=
            self.n_a, axis=-1), axis=-2)
        x_pos = (x_ / self.a_interval).round_().clip_(min=0, max=self.
            a_qmax - 1).multiply_(y=paddle.to_tensor(self.a_interval))
        x_neg = (x_ / self.a_neg_interval).round_().clip_(min=-self.a_qmax,
            max=0).multiply_(y=paddle.to_tensor(self.a_neg_interval))
        return (x_pos + x_neg).reshape(x.shape)

    def _search_best_a_interval(self, input_interval_candidates=None):
        tmp_a_interval = self.a_interval.unsqueeze(axis=-1)
        for a in range(self.n_a):
            batch_similarities = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x = self.raw_input[b_st:b_ed].cuda(blocking=True)
                raw_out_expanded = self.raw_out[b_st:b_ed].cuda(blocking=True
                    ).unsqueeze(axis=-2)
                raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True) if self.raw_grad is not None else None
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                    cur_a_interval = tmp_a_interval.tile(repeat_times=[1, 1,
                        p_ed - p_st])
                    cur_a_interval[a:a + 1, :, :] = input_interval_candidates[a
                        :a + 1, :, p_st:p_ed]
                    w_sim, bias_sim = self.quant_weight_bias()
                    x_sim = paddle.concat(x=paddle.chunk(x=x.unsqueeze(axis
                        =-2), chunks=self.n_a, axis=-1), axis=-2).unsqueeze(
                        axis=-1)
                    x_pos = (x_sim / cur_a_interval).round_().clip_(min=0,
                        max=self.a_qmax - 1) * cur_a_interval
                    x_neg = (x_sim / self.a_neg_interval).round_().clip_(min
                        =-self.a_qmax, max=0) * self.a_neg_interval
                    x_sim = (x_pos + x_neg).transpose(perm=[*list(range(len(tuple(x_sim.shape)) - 3)), -1, -3, -2]).reshape(
                            (*tuple(x.shape)[:-1], p_ed - p_st, tuple(x.shape)[-1]))
                    out_sim = paddle.nn.functional.linear(x=x_sim, weight=
                        w_sim.T, bias=bias_sim)
                    similarity = self._get_similarity(raw_out_expanded,
                        out_sim, self.metric, raw_grad)
                    similarity = paddle.mean(x=similarity, axis=list(range(
                        1, len(tuple(similarity.shape)) - 1)))
                    similarity = paddle.sum(x=similarity, axis=0, keepdim=True)
                    similarities.append(similarity)
                similarities = paddle.concat(x=similarities, axis=1)
                batch_similarities.append(similarities)
            batch_similarities = paddle.concat(x=batch_similarities, axis=0
                ).sum(axis=0, keepdim=False)
            a_best_index = batch_similarities.argmax(axis=0, keepdim=True
                ).reshape([1, 1, -1]).cuda(blocking=True)
            tmp_a_interval[a:a + 1, :, :] = paddle.take_along_axis(arr=
                input_interval_candidates[a:a + 1, :, :], axis=2, indices=
                a_best_index, broadcast=False)
        self.a_interval = tmp_a_interval.squeeze(axis=-1)
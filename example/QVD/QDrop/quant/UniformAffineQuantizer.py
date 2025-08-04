import sys
import paddle
from typing import Union
import time
import warnings


class StraightThrough(paddle.nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: paddle.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(y=p).sum(axis=1).mean()
    else:
        return (pred - tgt).abs().pow(y=p).mean()


class UniformAffineQuantizer(paddle.nn.Layer):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int=8, symmetric: bool=False, channel_wise:
        bool=False, scale_method: str='minmax', leaf_param: bool=False,
        prob: float=1.0, in_channels=1, shape=1, per_batch=False,
        act_quantile=1.0, split_SD=False, s_nbits=None):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.s_nbits = n_bits if s_nbits is None else s_nbits
        self.n_levels = 2 ** self.n_bits
        self.s_n_levels = 2 ** self.s_nbits
        delta = paddle.to_tensor(data=1.0).tile(repeat_times=in_channels
            ).astype(dtype='float16')
        zero_point = paddle.to_tensor(data=0.0).tile(repeat_times=in_channels
            ).astype(dtype='float16')
        while delta.dim() < shape:
            delta = delta[:, (None)]
            zero_point = zero_point[:, (None)]
        """稠密矩阵的参数"""
        self.delta = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =delta)
        self.zero_point = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=zero_point)
        """稀疏矩阵的参数"""
        self.s_delta = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =delta)
        self.s_zero_point = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=zero_point)
        self.inited = True
        """if leaf_param, use EMA to set scale"""
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = paddle.to_tensor(data=1e-08, dtype='float32')
        self.act_quantile = act_quantile
        """mse params"""
        self.scale_method = scale_method
        self.one_side_dist = None
        self.num = 50
        """ per batch params"""
        self.per_batch = per_batch
        """for activation quantization"""
        self.running_min = None
        self.running_max = None
        self.sum_min = None
        self.sum_max = None
        self.s_sum_min = None
        self.s_sum_max = None
        self.sum_n = None
        self.split_SD = split_SD
        self.lower_percentile = None
        self.upper_percentile = None
        self.percentile_n = 0
        self.exist_outlier = None
        """do like dropout"""
        self.prob = prob
        self.is_training = False
        self.timewise_quant_mode = False
        self.fast_mode = False
        self.now_t = None
        self.timestep_params = {}

    def change_timestep(self, t):
        assert self.timewise_quant_mode
        assert type(t) == int
        if self.now_t is not None:
            self.timestep_params[self.now_t] = [self.delta, self.zero_point,
                self.inited]
        if t not in self.timestep_params:
            self.timestep_params[t] = [1, 0, False]
        self.delta, self.zero_point, self.inited = self.timestep_params[t]
        self.now_t = t

    def set_inited(self, inited: bool=True):
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def update_quantize_range_abs_avg(self, x_min, x_max):
        if self.sum_min is None or self.sum_min.dim() == 0:
            self.sum_min = paddle.zeros_like(x=x_max)
            self.sum_max = paddle.zeros_like(x=x_max)
            self.sum_n = 0
        try:
            self.sum_max += x_max.astype(dtype='float32')
            self.sum_min += x_min.astype(dtype='float32')
            self.sum_n += 1
        except:
            import pdb
            pdb.set_trace()
        return self.sum_min / self.sum_n, self.sum_max / self.sum_n

    def update_quantize_range_abs_avg_splitSD(self, s_min, s_max, d_min, d_max
        ):
        if self.sum_min is None:
            self.sum_min = paddle.zeros_like(x=d_max)
            self.sum_max = paddle.zeros_like(x=d_max)
            self.s_sum_max = paddle.zeros_like(x=d_max)
            self.s_sum_min = paddle.zeros_like(x=d_max)
            self.sum_n = 0
        self.sum_max += d_max.astype(dtype='float32')
        self.sum_min += d_min.astype(dtype='float32')
        self.s_sum_min += s_min.astype(dtype='float32')
        self.s_sum_max += s_max.astype(dtype='float32')
        self.sum_n += 1
        return (self.sum_min / self.sum_n, self.sum_max / self.sum_n, self.
            s_sum_min / self.sum_n, self.s_sum_max / self.sum_n)

    def update_percentile_range_abs_avg(self, lower_percentile,
        upper_percentile, act_quantile):
        if self.lower_percentile is None:
            self.lower_percentile = 0
            self.upper_percentile = 0
            self.percentile_n = 0
            self.act_quantile = 0
        self.lower_percentile += lower_percentile.astype(dtype='float32')
        self.upper_percentile += upper_percentile.astype(dtype='float32')
        self.act_quantile += act_quantile
        self.percentile_n += 1
        return (self.lower_percentile / self.percentile_n, self.
            upper_percentile / self.percentile_n, self.act_quantile / self.
            percentile_n)

    def forward(self, x: paddle.Tensor):
        if self.inited is False:
            if self.leaf_param:
                if self.exist_outlier_(x) and self.split_SD:
                    (self.s_delta, self.s_zero_point, self.delta, self.
                        zero_point, S, D) = (self.init_quantization_scale(x
                        .clone().detach(), self.channel_wise))
                else:
                    self.delta, self.zero_point = self.init_quantization_scale(
                        x.clone().detach(), self.channel_wise)
            elif self.exist_outlier_(x) and self.split_SD:
                (self.s_delta, self.s_zero_point, self.delta, self.
                    zero_point, S, D) = (self.init_quantization_scale(x.
                    clone().detach(), self.channel_wise))
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x
                    .clone().detach(), self.channel_wise)
        if self.fast_mode:
            x_sim = (x / self.delta).round_().clip_(min=-self.zero_point,
                max=self.n_levels - 1 - self.zero_point).multiply_(y=paddle
                .to_tensor(self.delta))
            return x_sim
        if self.split_SD and self.exist_outlier_(x):
            if self.inited is True:
                lower_percentile, upper_percentile = self.batch_quantile(x.
                    astype(dtype='float32'), (1.0 - self.act_quantile /
                    self.percentile_n) / 2, batch_size=int(x.size / 32))
                D = x.clone()
                D[D < lower_percentile] = 0
                D[D > upper_percentile] = 0
                S = x.clone()
                S[(x >= lower_percentile) & (x <= upper_percentile)] = 0
            s_int = round_ste(S / self.s_delta) + self.s_zero_point
            s_quant = paddle.clip(x=s_int, min=0, max=self.s_n_levels - 1)
            s_dequant = (s_quant - self.s_zero_point) * self.s_delta
            d_int = round_ste(D / self.delta) + self.zero_point
            d_quant = paddle.clip(x=d_int, min=0, max=self.n_levels - 1)
            d_dequant = (d_quant - self.zero_point) * self.delta
            x_dequant = d_dequant + s_dequant
        elif self.is_per_batch(x):
            repeat_time = [1] * len(tuple(x.shape))
            repeat_time[0] = int(tuple(x.shape)[0] / 16)
            delta = self.delta.tile(repeat_times=repeat_time)
            zero_point = self.zero_point.tile(repeat_times=repeat_time)
            x_int = round_ste(x / delta) + zero_point
            x_quant = paddle.clip(x=x_int, min=0, max=self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta
        else:
            x_int = round_ste(x / self.delta) + self.zero_point
            x_quant = paddle.clip(x=x_int, min=0, max=self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = paddle.where(condition=paddle.rand(shape=x.shape, dtype
                =x.dtype) < self.prob, x=x_dequant, y=x)
        else:
            x_ans = x_dequant
        return x_ans

    def exist_outlier_(self, x):
        self.exist_outlier = True
        if self.exist_outlier is not None:
            return self.exist_outlier
        lowerq, upperq = self.batch_quantile(x.astype(dtype='float32'), 
            0.001, int(x.size / 32))
        all_range = x.max() - x.min()
        main_range = upperq - lowerq
        self.exist_outlier = all_range / main_range >= 2.7
        return self.exist_outlier

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(y=p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = paddle.flatten(x=x, start_axis=1)
            return y.mean(axis=1)

    def calculate_qparams(self, min_val, max_val, is_S=False):
        # one_dim or one element
        n_levels = self.s_n_levels if is_S else self.n_levels
        quant_min, quant_max = 0, n_levels - 1
        min_val_neg = paddle.minimum(min_val, paddle.zeros_like(min_val))
        max_val_pos = paddle.maximum(max_val, paddle.zeros_like(max_val))

        scale = (max_val_pos.astype('float32') - min_val_neg.astype('float32')) / float(quant_max - quant_min)
        scale = paddle.maximum(scale, self.eps)
        zero_point = quant_min - paddle.round(min_val_neg / scale)
        zero_point = paddle.clip(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x, x_max, x_min, is_S=False):
        delta, zero_point = self.calculate_qparams(x_min, x_max, is_S)
        n_levels = self.s_n_levels if is_S else self.n_levels
        
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)

        if self.leaf_param and self.is_per_batch(x):
            new_shape = [1] * len(x.shape)
            new_shape[0] = 16
            repeat_time = [1] * len(x.shape)
            repeat_time[0] = int(x.shape[0] / 16)
            delta = delta.reshape(new_shape).tile(repeat_time)
            zero_point = zero_point.reshape(new_shape).tile(repeat_time)
        
        x_int = paddle.round(x / delta)
        x_quant = paddle.clip(x_int + zero_point, 0, n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = paddle.flatten(x, start_axis=1)
            x_min, x_max = paddle.min(y, dim=1), paddle.max(y, dim=1)
            # may also have the one side distribution in some channels
            x_max = paddle.maximum(x_max, paddle.zeros_like(x_max))
            x_min = paddle.minimum(x_min, paddle.zeros_like(x_min))
        else:
            x_min, x_max = paddle.min(x), paddle.max(x)

        xrange = x_max - x_min
        best_score = paddle.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()

        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = paddle.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = paddle.where(score < best_score, new_min, best_min)
                best_max = paddle.where(score < best_score, new_max, best_max)
                best_score = paddle.minimum(best_score, score)
        return best_min, best_max
    
    def perform_1D_search(self, x, is_S=False):
        if self.is_per_batch(x):
            x_min, x_max = None, None
            splits_nums = x.shape[0] // 16
            for i in range(splits_nums):
                y = paddle.flatten(x, 1)[i*16:(i+1)*16, :]
                if x_min is None:
                    x_min, x_max = paddle.min(y, 1), paddle.max(y, 1)
                else:
                    min_, max_ = paddle.min(y, 1), paddle.max(y, 1)
                    x_min = 0.5 * min_ + 0.5 * x_min
                    x_max = 0.5 * max_ + 0.5 * x_max
        else:
            if self.channel_wise:
                y = paddle.flatten(x, 1)
                x_min, x_max = paddle.min(y, 1), paddle.max(y, 1)
            else:
                x_min, x_max = paddle.min(x), paddle.max(x)

        xrange = paddle.maximum(x_min.abs(), x_max)
        best_score = paddle.full_like(x_min, 1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()

        if not self.channel_wise and not self.is_per_batch(x):
            thres = (xrange / self.num) * paddle.arange(1, self.num + 1, dtype=x.dtype)
            new_min = paddle.zeros_like(thres) if self.one_side_dist == "pos" else -thres
            new_max = paddle.zeros_like(thres) if self.one_side_dist == "neg" else thres
            n_levels = self.s_n_levels if is_S else self.n_levels
            quant_min, quant_max = 0, n_levels - 1
            scale = (new_max - new_min) / (quant_max - quant_min)
            scale = paddle.maximum(scale, self.eps)
            zero_point = -paddle.round(new_min / scale)
            zero_point = paddle.clip(zero_point, quant_min, quant_max).reshape((-1, 1))
            scale = scale.reshape((-1, 1))
            
            scores = []
            batch = 16
            for i in range(0, self.num, batch):
                x_int = (x.reshape((1, -1)) / scale[i:i + batch]).round()
                x_int = paddle.clip(x_int, -zero_point[i:i + batch], n_levels - 1 - zero_point[i:i + batch])
                x_sim = x_int * scale[i:i + batch]
                score = paddle.abs(x_sim - x.reshape((1, -1))) ** 2.4
                score = paddle.mean(score, axis=1)
                scores.append(score)

            ind = paddle.argmin(paddle.concat(scores))
            best_min1 = new_min[ind]
            best_max1 = new_max[ind]
            return best_min1, best_max1

        for i in range(1, self.num + 1):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))

            x_q = self.quantize(x, new_max, new_min, is_S)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = paddle.where(score < best_score, new_min, best_min)
            best_max = paddle.where(score < best_score, new_max, best_max)
            best_score = paddle.minimum(score, best_score)

        return best_min, best_max

    def perform_1D_search_qdiff(self, x, is_S=False):
        if self.is_per_batch(x):
            x_min, x_max = None, None
            splits_nums = x.shape[0] // 16
            for i in range(splits_nums):
                y = paddle.flatten(x, 1)[i * 16:(i + 1) * 16, :]
                if x_min is None:
                    x_min, x_max = paddle.min(y, 1), paddle.max(y, 1)
                else:
                    min_, max_ = paddle.min(y, 1), paddle.max(y, 1)
                    x_min = 0.5 * min_ + 0.5 * x_min
                    x_max = 0.5 * max_ + 0.5 * x_max
        else:
            if self.channel_wise:
                y = paddle.flatten(x, 1)
                x_min, x_max = paddle.min(y, 1), paddle.max(y, 1)
            else:
                x_min, x_max = paddle.min(x), paddle.max(x)

        best_score = paddle.full_like(x_min, 1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        n_levels = self.s_n_levels if is_S else self.n_levels

        if not self.channel_wise and not self.is_per_batch(x):
            new_min = x_min * (1.0 - (paddle.arange(1, self.num + 1, dtype=x.dtype, device=x.place) * 0.01))
            new_max = x_max * (1.0 - (paddle.arange(1, self.num + 1, dtype=x.dtype, device=x.place) * 0.01))

            quant_min, quant_max = 0, n_levels - 1
            scale = (new_max - new_min) / (quant_max - quant_min)
            scale = paddle.maximum(scale, self.eps)
            zero_point = -paddle.round(new_min / scale)
            zero_point = paddle.clip(zero_point, quant_min, quant_max).reshape((-1, 1))
            scale = scale.reshape((-1, 1))

            scores = []
            batch = 16
            for i in range(0, self.num, batch):
                x_int = (x.reshape((1, -1)) / scale[i:i + batch]).round()
                x_int = paddle.clip(x_int, -zero_point[i:i + batch], n_levels - 1 - zero_point[i:i + batch])
                x_sim = x_int * scale[i:i + batch]
                score = paddle.abs(x_sim - x.reshape((1, -1))) ** 2.4
                score = paddle.mean(score, axis=1)
                scores.append(score)

            ind = paddle.argmin(paddle.concat(scores))
            best_min1 = new_min[ind]
            best_max1 = new_max[ind]
            return best_min1, best_max1

        for i in range(1, self.num + 1):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))

            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = paddle.where(score < best_score, new_min, best_min)
            best_max = paddle.where(score < best_score, new_max, best_max)
            best_score = paddle.minimum(score, best_score)

        return best_min, best_max


    def perform_minmax_search(self, x):
        if self.is_per_batch(x):
            x_min, x_max = None, None
            splits_nums = x.shape[0] // 16
            for i in range(splits_nums):
                y = paddle.flatten(x, 1)[i * 16:(i + 1) * 16, :]
                if x_min is None:
                    x_min, x_max = paddle.min(y, 1), paddle.max(y, 1)
                else:
                    min_, max_ = paddle.min(y, 1), paddle.max(y, 1)
                    x_min = 0.5 * min_ + 0.5 * x_min
                    x_max = 0.5 * max_ + 0.5 * x_max
            new_shape = [1] * len(x.shape)
            new_shape[0] = 16
            x_min = paddle.reshape(x_min, new_shape)
            x_max = paddle.reshape(x_max, new_shape)
        else:
            if self.channel_wise:
                y = paddle.flatten(x, 1)
                x_min, x_max = paddle.min(y, 1), paddle.max(y, 1)
            else:
                x_min, x_max = paddle.min(x), paddle.max(x)

        return x_min.clone(), x_max.clone()
    
    def perform_quantile_search(self, x):
        start, end = 0.8, 0.9999
        step = (end - start) / 25
        S = paddle.zeros_like(x=x)
        D = paddle.zeros_like(x=x)
        quantile = 0.8
        best_loss = paddle.to_tensor(data=[10000000000.0], dtype='float32').to(
            'cuda')
        best_upper, best_lower = None, None
        best_edge = None
        for i in range(26):
            q = start + i * step
            lower_percentile, upper_percentile = self.batch_quantile(x.
                astype(dtype='float32'), (1.0 - q) / 2, batch_size=int(x.
                size / 32))
            S.zero_()
            D.zero_()
            d_mask = (x >= lower_percentile) & (x <= upper_percentile)
            s_mask = (x < lower_percentile) | (x > upper_percentile)
            D[d_mask] = x[d_mask]
            S[s_mask] = x[s_mask]
            min_S, max_S = self.perform_minmax_search(S)
            min_D, max_D = self.perform_minmax_search(D)
            s_quant = self.quantize(S, max_S, min_S, is_S=True)
            d_quant = self.quantize(D, max_D, min_D)
            lossS = self.lp_loss(S.astype(dtype='float32'), s_quant.astype(
                dtype='float32'), 2.4)
            lossD = self.lp_loss(D.astype(dtype='float32'), d_quant.astype(
                dtype='float32'), 2.4)
            loss = lossS + lossD
            if paddle.sum(x=best_loss) > paddle.sum(x=loss):
                best_loss = loss
                quantile = q
                best_upper = upper_percentile
                best_lower = lower_percentile
                best_edge = (1.0 - q) / 2
        print(f'best_quantile:{quantile}')
        return quantile

    def get_x_min_x_max(self, x, is_S=False):
        if 'mse' in self.scale_method:
            if self.one_side_dist is None:
                self.one_side_dist = 'pos' if x.min(
                    ) >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
            if self.one_side_dist != 'no' or self.sym:
                if 'qdiff' in self.scale_method:
                    best_min, best_max = self.perform_1D_search_qdiff(x, is_S)
                else:
                    best_min, best_max = self.perform_1D_search(x, is_S)
            else:
                best_min, best_max = self.perform_2D_search(x)
        elif 'minmax' in self.scale_method:
            best_min, best_max = self.perform_minmax_search(x)
        else:
            raise NotImplementedError
        if self.leaf_param:
            if not self.split_SD or not self.exist_outlier:
                return self.update_quantize_range_abs_avg(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel_splitSD(self, x: paddle.Tensor):
        if self.split_SD:
            with paddle.no_grad():
                act_quantile = self.perform_quantile_search(x)
                _, _, act_quantile = self.update_percentile_range_abs_avg(
                    paddle.to_tensor(data=[0.0], dtype='float32'), paddle.
                    to_tensor(data=[0.0], dtype='float32'), act_quantile)
                print(f'avg quantile :{act_quantile}')
                edge = (1.0 - act_quantile) / 2.0
                lower_percentile, upper_percentile = self.batch_quantile(x.
                    astype(dtype='float32'), edge, batch_size=int(x.size / 32))
                D = x.clone()
                D[D < lower_percentile] = 0
                D[D > upper_percentile] = 0
                S = x.clone()
                S[(x >= lower_percentile) & (x <= upper_percentile)] = 0
                S_min, S_max = self.get_x_min_x_max(S, is_S=True)
                D_min, D_max = self.get_x_min_x_max(D)
                D_min, D_max, S_min, S_max = (self.
                    update_quantize_range_abs_avg_splitSD(S_min, S_max,
                    D_min, D_max))
                S_scale, S_zero_point = self.calculate_qparams(S_min, S_max,
                    is_S=True)
                D_scale, D_zero_point = self.calculate_qparams(D_min, D_max)
        return S_scale, S_zero_point, D_scale, D_zero_point, S, D

    def init_quantization_scale_channel(self, x: paddle.Tensor):
        with paddle.no_grad():
            x_min, x_max = self.get_x_min_x_max(x)
            scale, zero_point = self.calculate_qparams(x_min, x_max)
        return scale, zero_point

    def init_quantization_scale(self, x_clone: paddle.Tensor, channel_wise:
        bool=False):
        if self.is_per_batch(x_clone):
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(tuple(x_clone.shape))
            new_shape[0] = 16
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        elif self.split_SD and self.exist_outlier_(x_clone):
            S_scale, S_zero_point, D_scale, D_zero_point, S, D = (self.
                init_quantization_scale_channel_splitSD(x_clone))
            if channel_wise:
                new_shape = [1] * len(tuple(x_clone.shape))
                new_shape[0] = tuple(x_clone.shape)[0]
                S_scale = S_scale.reshape(new_shape)
                S_zero_point = S_zero_point.reshape(new_shape)
                D_scale = D_scale.reshape(new_shape)
                D_zero_point = D_zero_point.reshape(new_shape)
            return paddle.base.framework.EagerParamBase.from_tensor(tensor=
                S_scale), paddle.base.framework.EagerParamBase.from_tensor(
                tensor=S_zero_point
                ), paddle.base.framework.EagerParamBase.from_tensor(tensor=
                D_scale), paddle.base.framework.EagerParamBase.from_tensor(
                tensor=D_zero_point), S, D
        elif channel_wise:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(tuple(x_clone.shape))
            new_shape[0] = tuple(x_clone.shape)[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
            # print(f"x_clone shape: {tuple(x_clone.shape)}")
            # print(f"zero_point shape: {tuple(zero_point.shape)}")
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return paddle.base.framework.EagerParamBase.from_tensor(tensor=delta
            ), paddle.base.framework.EagerParamBase.from_tensor(tensor=
            zero_point)

    def is_per_batch(self, x):
        return self.per_batch and (tuple(x.shape)[0] == 32 or tuple(x.shape
            )[0] == 16)

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 32, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def batch_quantile(self, tensor, q, batch_size):
        n = tensor.size
        tensor = tensor.reshape(-1)
        num_batches = (n + batch_size - 1) // batch_size
        lower_quantiles = []
        upper_quantiles = []
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, n)
            batch = tensor[start:end]
            lower_quantiles.append(paddle.quantile(x=batch, q=q))
            upper_quantiles.append(paddle.quantile(x=batch, q=1.0 - q))
        lower_quantiles_tensor = paddle.to_tensor(data=lower_quantiles)
        upper_quantiles_tensor = paddle.to_tensor(data=upper_quantiles)
        overall_lower_quantile = paddle.quantile(x=lower_quantiles_tensor, q=q)
        overall_upper_quantile = paddle.quantile(x=upper_quantiles_tensor,
            q=1.0 - q)
        return overall_lower_quantile, overall_upper_quantile


    def extra_repr(self):
        return "bit={}, is_training={}, inited={}".format(
            self.n_bits, self.is_training, self.inited
        )
import numpy as np
import paddle
import paddle.nn as nn

from .base import BaseObserver


class PercentileObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode,
                 percentile_sigma=0.01, percentile_alpha=0.99999):
        super(PercentileObserver, self).__init__(
            module_type, bit_type, calibration_mode)
        self.percentile_sigma = percentile_sigma
        self.percentile_alpha = percentile_alpha
        self.symmetric = self.bit_type.signed

    def update(self, v):
        # channel-wise needs too much time.
        assert self.calibration_mode == "layer_wise"
        v = self.reshape_tensor(v)
        try:
            cur_max = paddle.quantile(v.flatten(), self.percentile_alpha)
            cur_min = paddle.quantile(v.flatten(), 1.0 - self.percentile_alpha)
        except:
            cur_max = paddle.to_tensor(np.percentile(
                v.flatten().cpu(), self.percentile_alpha * 100), dtype=paddle.float32)
            cur_min = paddle.to_tensor(np.percentile(
                v.flatten().cpu(), (1 - self.percentile_alpha) * 100), dtype=paddle.float32)

        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + \
                self.percentile_sigma * (cur_max - self.max_val)

        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + \
                self.percentile_sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = paddle.ones_like(max_val, dtype=paddle.float32)
        zero_point = paddle.zeros_like(max_val, dtype=paddle.int64)

        if self.symmetric:
            max_val = paddle.maximum(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale = paddle.clip(scale, min=self.eps)
            zero_point = paddle.zeros_like(max_val, dtype=paddle.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = paddle.clip(scale, min=self.eps)
            zero_point = qmin - paddle.round(min_val / scale)
            zero_point = paddle.clip(zero_point, min=qmin, max=qmax)

        return scale, zero_point

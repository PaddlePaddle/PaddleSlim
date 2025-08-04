import paddle
from .base import BaseObserver

class EmaObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode, ema_sigma=0.01):
        super(EmaObserver, self).__init__(module_type, bit_type, calibration_mode)
        self.ema_sigma = ema_sigma
        self.symmetric = self.bit_type.signed

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = paddle.max(v, axis=1)
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + self.ema_sigma * (cur_max - self.max_val)
        cur_min = paddle.min(v, axis=1)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + self.ema_sigma * (cur_min - self.min_val)

        if self.calibration_mode == "layer_wise":
            self.max_val = paddle.max(self.max_val)
            self.min_val = paddle.min(self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = paddle.ones_like(max_val, dtype='float32')
        zero_point = paddle.zeros_like(max_val, dtype='int64')

        if self.symmetric:
            max_val = paddle.maximum(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale = paddle.clip(scale, min=self.eps)
            zero_point = paddle.zeros_like(max_val, dtype='int64')
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = paddle.clip(scale, min=self.eps)
            zero_point = qmin - paddle.round(min_val / scale)
            zero_point = paddle.clip(zero_point, min=qmin, max=qmax)
        
        return scale, zero_point


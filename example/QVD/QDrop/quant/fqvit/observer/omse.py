import paddle
from .base import BaseObserver
from .utils import lp_loss

class OmseObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode):
        super(OmseObserver, self).__init__(
            module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = paddle.max(v, axis=1)
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = paddle.maximum(cur_max, self.max_val)
        cur_min = paddle.min(v, axis=1)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = paddle.minimum(cur_min, self.min_val)

        if self.calibration_mode == "layer_wise":
            self.max_val = paddle.max(self.max_val)
            self.min_val = paddle.min(self.min_val)

    def get_quantization_params(self, inputs):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        for i in range(90):
            new_max = max_val * (1.0 - (i * 0.01))
            new_min = min_val * (1.0 - (i * 0.01))
            new_scale = (new_max - new_min) / float(qmax - qmin)
            new_scale = paddle.clip(new_scale, min=self.eps)
            new_zero_point = qmin - paddle.round(new_min / new_scale)
            new_zero_point = paddle.clip(new_zero_point, min=qmin, max=qmax)
            inputs_q = ((inputs / new_scale + new_zero_point).round().clip(qmin,
                        qmax) - new_zero_point) * new_scale
            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            score = lp_loss(inputs, inputs_q, p=2.0, reduction="all")
            if score < best_score:
                best_score = score
                self.max_val = new_max
                self.min_val = new_min
                scale = new_scale
                zero_point = new_zero_point
        return scale, zero_point

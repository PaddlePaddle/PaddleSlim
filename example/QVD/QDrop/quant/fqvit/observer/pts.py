import paddle
from .base import BaseObserver
from .utils import lp_loss


class PtsObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtsObserver, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)  # 假设 reshape_tensor 是自定义方法
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

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = paddle.max(max_val)
        min_val_t = paddle.min(min_val)
        
        scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8 = paddle.clip(scale8, min=self.eps)  # 类似 clamp_
        
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        
        zero_point = qmin - paddle.round(min_val_t / scale8)
        zero_point = paddle.clip(zero_point, min=qmin, max=qmax)  # 类似 clamp_
        
        scale_mask = paddle.ones_like(max_val)
        
        for j in range(inputs.shape[2]):
            data = inputs[..., j].unsqueeze(-1)
            
            # Quantization for different scales
            data_q1 = ((data / scale1 + zero_point).round().clip(min=qmin, max=qmax) - zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clip(min=qmin, max=qmax) - zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clip(min=qmin, max=qmax) - zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clip(min=qmin, max=qmax) - zero_point) * scale8
            
            # Loss computation for each scale
            score1 = lp_loss(data, data_q1, p=2.0, reduction="all")
            score2 = lp_loss(data, data_q2, p=2.0, reduction="all")
            score4 = lp_loss(data, data_q4, p=2.0, reduction="all")
            score8 = lp_loss(data, data_q8, p=2.0, reduction="all")
            
            score = [score1, score2, score4, score8]
            scale_mask[j] *= 2 ** score.index(min(score))  # Choose best scale

        scale = scale1 * scale_mask
        return scale, zero_point

import paddle
from .base import BaseObserver
from .utils import lp_loss


class TwinObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode):
        super(TwinObserver, self).__init__(module_type, bit_type, calibration_mode)

        self.a_qmax = bit_type.upper_bound
        self.a_neg_interval = 0.16997124254703522 / self.a_qmax
        self.a_interval = None
        self.ema_sigma = 0.01

    def update(self, x):
        x = self.reshape_tensor(x)  # 假设reshape_tensor是自定义方法
        cur_max = paddle.max(x, axis=1)
        cur_a_interval = (paddle.abs(x).max() / (self.a_qmax - 0.5)).detach()
        if self.a_interval is None:
            self.a_interval = cur_a_interval
        else:
            self.a_interval = self.a_interval + \
                self.ema_sigma * (cur_a_interval - self.a_interval)

    def get_quantization_params(self, inputs, *args, **kwargs):
        return self.a_interval, self.a_neg_interval

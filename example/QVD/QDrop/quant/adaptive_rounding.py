import paddle
from .quant_layer import UniformAffineQuantizer, round_ste


class AdaRoundQuantizer(paddle.nn.Layer):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: paddle.
        Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = paddle.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = paddle.floor(x=x / self.delta)
            rest = x / self.delta - x_floor
            x_int = x_floor + paddle.bernoulli(x=rest)
            print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = paddle.floor(x=x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).astype(dtype='float32')
        else:
            raise ValueError('Wrong rounding mode')
        x_quant = paddle.clip(x=x_int + self.zero_point, min=0, max=self.
            n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta
        return x_float_q

    def get_soft_targets(self):
        return paddle.clip(x=paddle.nn.functional.sigmoid(x=self.alpha) * (
            self.zeta - self.gamma) + self.gamma, min=0, max=1)

    def init_alpha(self, x: paddle.Tensor):
        x_floor = paddle.floor(x=x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = x / self.delta - x_floor
            alpha = -paddle.log(x=(self.zeta - self.gamma) / (rest - self.
                gamma) - 1)
            self.alpha = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=alpha)
        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'bit={}'.format(self.n_bits)

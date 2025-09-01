import paddle
from paddle import nn
from quantizers.uniform import UniformQuantizer
from quantizers._ste import round_ste
import paddle.nn.functional as F


class AdaRoundQuantizer(nn.Layer):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uq: UniformQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uq: UniformQuantizer, weight_tensor: paddle.Tensor, round_mode='learned_hard_sigmoid'):
        super().__init__()
        # copying all attributes from UniformQuantizer
        self.n_bits = uq.n_bits
        self.n_levels = uq.n_levels
        self.channel_wise = uq.channel_wise
        self.sym = uq.sym
        #scale=paddle.to_tensor(uq.scale)
        self.scale = paddle.create_parameter(shape=uq.scale.shape,
                        dtype=uq.scale.dtype,
                        default_initializer=paddle.nn.initializer.Assign(uq.scale))
        self.zero_point = paddle.create_parameter(shape=uq.zero_point.shape,
                        dtype=uq.zero_point.dtype,
                        default_initializer=paddle.nn.initializer.Assign(uq.zero_point))
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = paddle.round(x / self.scale)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.scale)
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = paddle.floor(x / self.scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).astype('float32')
        else:
            raise ValueError('Wrong rounding mode')
        if self.sym:
            x_quant = paddle.clip(x_int, -self.n_levels, self.n_levels - 1)
            x_float_q = x_quant * self.scale
        else:
            x_quant = paddle.clip(x_int + self.zero_point, 0, 2 * self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * self.scale
        return x_float_q

    def get_soft_targets(self):
        return paddle.clip(F.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: paddle.Tensor):
        x_floor = paddle.floor(x / self.scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
            alpha = -paddle.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            alpha=paddle.to_tensor(alpha)
            self.alpha =paddle.create_parameter(shape=alpha.shape,
                        dtype=alpha.dtype,
                        default_initializer=paddle.nn.initializer.Assign(alpha))
        else:
            raise NotImplementedError

    def get_hard_value(self, x):
        init_shape = x.shape
        return ((paddle.floor(x.reshape(self.alpha.shape) / self.scale) + (self.alpha >= 0).astype('float32')) * self.scale).reshape(init_shape)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, round_mode={self.round_mode})'

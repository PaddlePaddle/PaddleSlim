import sys
import paddle
import os
import sys
sys.path.append('/mnt/disk2/charles/template/QVD-paddle/')
from tqdm import tqdm
from .UniformAffineQuantizer import UniformAffineQuantizer
from .fqvit.bit_type import BIT_TYPE_DICT
from .fqvit.observer.build import *
from .fqvit.quantizer.build import *

class QIntSoftmax(paddle.nn.Layer):

    def __init__(self, log_i_softmax=False, quant=False, calibrate=False,
        last_calibrate=False, bit_type=BIT_TYPE_DICT['int8'],
        calibration_mode='layer_wise', observer_str='minmax', quantizer_str
        ='uniform'):
        super(QIntSoftmax, self).__init__()
        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
            self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
            self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = x - 2 ** big >= 2 ** (big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.0]
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = paddle.floor(x=coef[1] / scaling_factor)
            c_int = paddle.floor(x=coef[2] / scaling_factor ** 2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor ** 2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931
            n = 30
            x0_int = paddle.floor(x=x0 / scaling_factor)
            x_int = paddle.max(x_int, n * x0_int)
            q = paddle.floor(x=x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = paddle.clip(x=paddle.floor(x=exp_int * 2 ** (n - q)),
                min=0)
            scaling_factor = exp_scaling_factor / 2 ** n
            return exp_int, scaling_factor
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(axis=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = paddle.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2 ** self.bit_type.bits
            qlog = paddle.clip(x=rounds, min=0, max=2 ** self.bit_type.bits - 1
                )
            deq_softmax = 2 ** -qlog
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = paddle.nn.functional.softmax(x, axis=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


class LogQuantizer(UniformAffineQuantizer):

    def __init__(self, n_bits: int=8, symmetric: bool=False, channel_wise:
        bool=False, scale_method: str='minmax', leaf_param: bool=False,
        prob: float=1.0, in_channels=1, shape=1, per_batch=False,
        act_quantile=1.0, split_SD=False, s_nbits=None):
        super(LogQuantizer, self).__init__(n_bits=n_bits, symmetric=
            symmetric, channel_wise=channel_wise, scale_method=scale_method,
            leaf_param=leaf_param, prob=prob, in_channels=in_channels,
            shape=shape, per_batch=per_batch, act_quantile=act_quantile,
            split_SD=split_SD, s_nbits=s_nbits)
        self.alpha = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.to_tensor(data=[1.0]))
        self.absmax = None

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = x - 2 ** big >= 2 ** (big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    def forward(self, x):
        x = x.astype(dtype='float32')
        if self.inited is False:
            self.absmax = self.init_quantization_scale(x)
        sign = paddle.sign(x=x)
        rounds = 0.0 - self.log_round(paddle.abs(x=x) / self.absmax)
        mask = rounds >= 2 ** self.n_bits
        qlog = paddle.clip(x=rounds, min=0, max=2 ** self.n_bits - 1)
        deq_x = 2 ** -qlog * sign * self.absmax
        deq_x[mask] = 0
        return deq_x

    def init_quantization_scale(self, x):
        if self.absmax is None:
            self.absmax = paddle.max(x=paddle.abs(x=x))
            return self.absmax
        return paddle.max(paddle.max(x=paddle.abs(x=x)), self.absmax)


class MixLogUniformQuanizer(paddle.nn.Layer):

    def __init__(self, n_bits: int=8, symmetric: bool=False, channel_wise:
        bool=False, scale_method: str='minmax', leaf_param: bool=True, prob:
        float=1.0, in_channels=1, shape=1, per_batch=False, act_quantile=
        1.0, split_SD=False, s_nbits=None) ->None:
        super(MixLogUniformQuanizer, self).__init__()
        self.log_quantizer = LogQuantizer(n_bits - 1, symmetric,
            channel_wise, scale_method, leaf_param, prob, in_channels,
            shape, per_batch, act_quantile, split_SD, s_nbits)
        self.uniform_quantizer = UniformAffineQuantizer(n_bits - 1,
            symmetric, channel_wise, 'mse', leaf_param, prob, in_channels,
            shape, per_batch, act_quantile, split_SD, s_nbits)
        self.persentile = 0.8

    def set_persentile(self, persentile):
        self.persentile = persentile

    def forward(self, x, edge):
        mask = x <= edge
        log_part = x.clone()
        uniform_part = x.clone()
        log_part[~mask] = 0.0
        uniform_part[mask] = paddle.max(x=x)
        dq_logpart = self.log_quantizer(log_part)
        dq_uniform_part = self.uniform_quantizer(uniform_part)
        dq_uniform_part[mask] = 0
        out = dq_logpart + dq_uniform_part
        return out


if __name__ == '__main__':

    paddle.seed(1)
    dir = (
        '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-platcv/tsl/code/outputs/TED/act_full/act_5bXiuSrZUKg#018047#018212_input'
        )
    time_embs = []
    for i in range(25):
        name = f'ResnetBlock3D_ins1_fwd{i}_time_emb_proj_in.pth'
        t_path = os.path.join(dir, name)
        t = paddle.load(path=str(t_path))
        time_embs.append(t.astype(dtype='float32'))
    n_bits = 6
    logQ = LogQuantizer(n_bits=n_bits)
    mixQ = MixLogUniformQuanizer(n_bits=n_bits, symmetric=True)
    logQ.set_inited(False)
    mixQ.log_quantizer.set_inited(False)
    mixQ.uniform_quantizer.set_inited(False)
    start_q = 0.5
    best_q = 0.5
    n = 100
    step = (4.421875 - start_q) / n
    best_loss = 10000000000.0
    absmax = 0
    best_s = None
    best_z = None
    with paddle.no_grad():
        for i in range(n + 1):
            q = start_q + i * step
            loss = 0
            for t in time_embs:
                mixQ(t, q)
            mixQ.uniform_quantizer.set_inited(True)
            mixQ.log_quantizer.set_inited(True)
            for t in time_embs:
                mix_dq_t = mixQ(t, q)
                loss += mixQ.uniform_quantizer.lp_loss(mix_dq_t, t)
            mixQ.uniform_quantizer.set_inited(False)
            mixQ.log_quantizer.set_inited(False)
            print(loss)
            if loss < best_loss:
                best_loss = loss
                best_q = q
                absmax = mixQ.log_quantizer.absmax.clone()
                best_s = mixQ.uniform_quantizer.delta.clone()
                best_z = mixQ.uniform_quantizer.zero_point.clone()
            mixQ.persentile = None
            mixQ.log_quantizer.absmax = None
            mixQ.uniform_quantizer.delta.zero_()
            mixQ.uniform_quantizer.zero_point.zero_()
            mixQ.uniform_quantizer.sum_max = None
            mixQ.uniform_quantizer.sum_min = None
    for t in time_embs:
        logQ(t)
    print(f'absmax = {absmax}, best_q={best_q}')
    mixQ.log_quantizer.absmax = absmax
    mixQ.uniform_quantizer.delta.data = best_s
    mixQ.uniform_quantizer.zero_point.data = best_z
    mixQ.log_quantizer.set_inited(True)
    mixQ.uniform_quantizer.set_inited(True)
    logQ.set_inited(True)
    loss_mix = 0
    loss_log = 0
    for t in time_embs:
        mix_dq_t = mixQ(t, best_q)
        log_dq_t = logQ(t)
        loss_log += logQ.lp_loss(log_dq_t, t)
        loss_mix += logQ.lp_loss(mix_dq_t, t)
    print(f'loss_log = {loss_log}')
    print(f'loss_mix = {loss_mix}')

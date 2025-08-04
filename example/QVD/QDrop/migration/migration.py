import sys
import paddle
import paddle.nn.functional as F
import logging

from ppdiffusers.models.attention_processor import Attention

from scipy.optimize import minimize_scalar
from .observer import MinMaxObserver
from .util_quant import fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine
from .util_layernorm import QuantizedLayerNorm

from magicanimate.models.diffusers_attention import  FeedForward
from magicanimate.models.orig_attention import CrossAttention as VersatileAttention
from QDrop.quant.quant_block import QuantLoRACompatibleLinear
logger = logging.getLogger('QVD')
shift_list = []
scale_list = []


def migration(act, layer, a_qconfig, w_qconfig, module_type, extra_dict=None):
    migrator = Migrator1DRangeSearch(act, layer, a_qconfig, w_qconfig,
        module_type, extra_dict)
    best_scale = migrator()
    scale_list.append(best_scale)
    shift_list.append(extra_dict['shift'])
    return best_scale


def fuse_migration(model):
    cnt = 0
    for name, module in model.named_sublayers():
        if isinstance(module, QuantizedLayerNorm):
            if cnt < len(shift_list):
                module.bias.data -= shift_list[cnt]
                module.weight.data /= scale_list[cnt]
                module.bias.data /= scale_list[cnt]
                cnt += 1


class MigratorBase(paddle.nn.Layer):

    def __init__(self, input, layer, a_qconfig, w_qconfig, module_type,
        extra_dict=None):
        super().__init__()
        self.input = input
        self.extra_dict = extra_dict
        self.a_qconfig = a_qconfig
        self.w_qconfig = w_qconfig
        self.module_type = module_type
        self.dtype = self.input.dtype
        self.device = self.input.place
        self.weight = self.get_weight(layer)
        self.layer = layer
        self.set_aminmax(input)
        abs_cmx = paddle.abs(x=self.cmx)
        abs_cmn = paddle.abs(x=self.cmn)
        self.absmx_perchannel = paddle.where(condition=abs_cmn > abs_cmx, x
            =abs_cmn, y=abs_cmx)
        self.amx = max(self.input.max(), paddle.to_tensor(data=0.0, dtype=
            self.dtype).to(self.device))
        self.amn = min(self.input.min(), paddle.to_tensor(data=0.0, dtype=
            self.dtype).to(self.device))
        logger.info('the module type is {}'.format(self.module_type))
        logger.info('the data type is {}, the device is {}'.format(self.
            dtype, self.device))
        logger.info('the activation range is {:.2f}, {:.2f}'.format(self.
            amn, self.amx))
        logger.info('the weight range is {:.2f}, {:.2f}'.format(self.weight
            .min(), self.weight.max()))
        self.output = self.get_output(self.input, self.weight)
        self.aob = MinMaxObserver(self.a_qconfig.bit, self.a_qconfig.
            symmetric, self.a_qconfig.ch_axis).to(self.device).to(self.dtype)
        self.wob = MinMaxObserver(self.w_qconfig.bit, self.w_qconfig.
            symmetric, self.w_qconfig.ch_axis).to(self.device).to(self.dtype)

    def set_aminmax(self, hidden_states):
        if self.module_type == 'conv':
            ori_shape = tuple(hidden_states.shape)
            self.cmn = (paddle.min(x=hidden_states.transpose(perm=[1, 0, 2,
                3]).reshape(ori_shape[1], -1), axis=1), paddle.argmin(x=
                hidden_states.transpose(perm=[1, 0, 2, 3]).reshape(
                ori_shape[1], -1), axis=1))[0].reshape(1, ori_shape[1], 1, 1)
            self.cmx = (paddle.max(x=hidden_states.transpose(perm=[1, 0, 2,
                3]).reshape(ori_shape[1], -1), axis=1), paddle.argmax(x=
                hidden_states.transpose(perm=[1, 0, 2, 3]).reshape(
                ori_shape[1], -1), axis=1))[0].reshape(1, ori_shape[1], 1, 1)
        else:
            self.cmn = self.input.min(0)[0].min(0)[0]
            self.cmx = self.input.max(0)[0].max(0)[0]

    def get_weight(self, layer):
        if isinstance(layer, Attention):
            wtok = layer.to_k.linear.weight.data
            wtoq = layer.to_q.linear.weight.data
            wtov = layer.to_v.linear.weight.data
            if layer.to_k.linear.bias is None:
                layer.to_k.linear.bias = (paddle.base.framework.
                    EagerParamBase.from_tensor(tensor=paddle.zeros(shape=
                    layer.inner_dim).to(next(layer.to_k.parameters()).place)))
                layer.to_q.linear.bias = (paddle.base.framework.
                    EagerParamBase.from_tensor(tensor=paddle.zeros(shape=
                    layer.inner_dim).to(next(layer.to_k.parameters()).place)))
                layer.to_v.linear.bias = (paddle.base.framework.
                    EagerParamBase.from_tensor(tensor=paddle.zeros(shape=
                    layer.inner_dim).to(next(layer.to_k.parameters()).place)))
            btok = layer.to_k.linear.bias.data
            btoq = layer.to_q.linear.bias.data
            btov = layer.to_v.linear.bias.data
            wtokqv = paddle.concat(x=[wtok, wtoq, wtov], axis=0)
            btokqv = paddle.concat(x=[btok, btoq, btov], axis=0)
            self.extra_dict['bias'] = btokqv
            self.scale = layer.scale
            return wtokqv
        elif isinstance(layer, VersatileAttention):
            wtok = layer.to_k.weight.data
            wtoq = layer.to_q.weight.data
            wtov = layer.to_v.weight.data
            if layer.to_k.bias is None:
                layer.to_k.bias = (paddle.base.framework.EagerParamBase.
                    from_tensor(tensor=paddle.zeros(shape=layer.inner_dim).
                    to(next(layer.to_k.parameters()).place)))
                layer.to_q.bias = (paddle.base.framework.EagerParamBase.
                    from_tensor(tensor=paddle.zeros(shape=layer.inner_dim).
                    to(next(layer.to_k.parameters()).place)))
                layer.to_v.bias = (paddle.base.framework.EagerParamBase.
                    from_tensor(tensor=paddle.zeros(shape=layer.inner_dim).
                    to(next(layer.to_k.parameters()).place)))
            btok = layer.to_k.bias.data
            btoq = layer.to_q.bias.data
            btov = layer.to_v.bias.data
            wtokqv = paddle.concat(x=[wtok, wtoq, wtov], axis=0)
            btokqv = paddle.concat(x=[btok, btoq, btov], axis=0)
            self.extra_dict['bias'] = btokqv
            self.scale = layer.scale
            return wtokqv
        elif isinstance(layer, QuantLoRACompatibleLinear):
            weight = layer.linear.weight
            bias = layer.linear.bias
            self.extra_dict['bias'] = bias
            return weight
        elif isinstance(layer, FeedForward):
            weight = layer.net[0].proj.linear.weight.data
            bias = layer.net[0].proj.linear.bias.data
            self.extra_dict['bias'] = bias
            return weight
        elif layer.name == 'conv2d' or layer.name == 'linear':
            weight = layer.weight.data
            bias = layer.bias.data
            self.extra_dict['bias'] = bias
            return weight
        else:
            raise NotImplementedError

    def get_output(self, input, weight):
        if self.module_type == 'qkv':
            output = self.qkv_function(input, weight)
        elif self.module_type == 'fc1':
            output = self.fc1_function(input, weight)
        elif self.module_type == 'conv':
            output = self.conv2d_function(input, weight)
        else:
            raise NotImplementedError
        return output

    def quantize(self, X, observer, clipping_range=None):
        org_shape = X.shape
        if clipping_range is not None:
            if 'Group' in self.a_qconfig['quantizer']:
                X = X.reshape([-1, self.a_qconfig['group_size']])
                min_val_cur, max_val_cur = observer(X)
            elif 'Token' in self.a_qconfig['quantizer']:
                X = X.reshape([-1, org_shape[-1]])
                min_val_cur, max_val_cur = observer(X)
            else:
                min_val_cur, max_val_cur = clipping_range
        else:
            if 'Group' in self.w_qconfig['quantizer']:
                X = X.reshape([-1, self.w_qconfig['group_size']])
            min_val_cur, max_val_cur = observer(X)
        
        scale, zp = observer.calculate_qparams(min_val_cur, max_val_cur)
        if observer.ch_axis == -1:
            X_q = fake_quantize_per_tensor_affine(
                X, scale.item(), zp.item(),
                observer.quant_min, observer.quant_max)
        else:
            X_q = fake_quantize_per_channel_affine(
                X, scale, zp, observer.ch_axis,
                observer.quant_min, observer.quant_max)
        X_q = X_q.reshape(org_shape)
        return X_q

    def get_qoutput(self, input, weight, clipping_range=None):
        qinput = self.quantize(input, self.aob, clipping_range)
        qweight = self.quantize(weight, self.wob)
        return self.get_output(qinput, qweight)

    def cac_scale(self, min_range, max_range):
        mx_scale = paddle.where(self.cmx > max_range, self.cmx / max_range, paddle.to_tensor(1.0, dtype=self.dtype).to(self.device))
        mn_scale = paddle.where(self.cmn < min_range, self.cmn / min_range, paddle.to_tensor(1.0, dtype=self.dtype).to(self.device))
        final_scale = paddle.maximum(mx_scale, mn_scale)
        return final_scale

    def get_best_scale(self, min_range, max_range):
        best_scale = self.cac_scale(min_range, max_range)
        logger.info('the best scale is {:.2f}, best min range is {:.2f}, \
            best max range is {:.2f}'.format(best_scale.max(), (self.input / best_scale).min(), (self.input / best_scale).max()))
        logger.info('the range of weight becomes {:.2f}, {:.2f}'.format((self.weight * best_scale).min(), (self.weight * best_scale).max()))
        return best_scale

    def loss_fx(self, pred, tgt, p=2.0):
        return paddle.abs(pred - tgt).pow(p).sum(-1).mean()

    def cac_loss(self, min_range, max_range):
        cur_scale = self.cac_scale(min_range, max_range)
        qoutput = self.get_qoutput(self.input / cur_scale, self.weight * cur_scale, (min_range, max_range))
        return self.loss_fx(qoutput, self.output)

    def qkv_function(self, input, weight):
        B, N, C = input.shape

        kqv = paddle.matmul(input, weight.T()) + self.extra_dict['bias']
        k, q, v = paddle.split(kqv, C, axis=-1)

        q = self.head_to_batch_dim(q).contiguous()
        k = self.head_to_batch_dim(k).contiguous()
        v = self.head_to_batch_dim(v).contiguous()


        attention_scores = paddle.matmul(q, k.transpose([0, 2, 1])) * self.scale


        attention_probs = paddle.nn.functional.softmax(attention_scores, axis=-1)

        output = paddle.matmul(attention_probs, v)
        return output.astype(paddle.float32)

    def conv2d_function(self, input, weight):
        return F.conv2d(input, weight, self.extra_dict['bias'], **self.extra_dict['conv_fwd_kwargs'])

    def fc1_function(self, input, weight):
        output = paddle.matmul(input, weight.T())
        return output.astype(paddle.float32)

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.extra_dict['num_heads']
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape([batch_size, seq_len, head_size, dim // head_size])
        tensor = tensor.transpose([0, 2, 1, 3])

        if out_dim == 3:
            tensor = tensor.reshape([batch_size * head_size, seq_len, dim // head_size])

        return tensor

    def forward(self):
        pass


class Migrator1DRangeSearch(MigratorBase):

    def __init__(self, input, weight, a_qconfig, w_qconfig, module_type,
        extra_dict=None):
        super().__init__(input, weight, a_qconfig, w_qconfig, module_type,
            extra_dict)
        self.num = max(100, int(self.amx / 0.5))

    def cac_scale_loss(self, mn_range, mx_range):
        return self.cac_loss(paddle.to_tensor(data=mn_range, dtype=self.
            dtype).to(self.device), paddle.to_tensor(data=mx_range, dtype=
            self.dtype).to(self.device))

    def search_migrate_range_1D(self):
        best_loss = None
        lower_bound = paddle.mean(x=self.absmx_perchannel)
        bounds = lower_bound, max(-self.amn.item(), self.amx.item())
        step = (bounds[1] - bounds[0]) / self.num
        mn_range = -bounds[1]
        mx_range = bounds[1]
        st = bounds[1]
        cnt = 0
        while st >= bounds[0]:
            loss = self.cac_scale_loss(-st, st)
            if best_loss is None or best_loss > loss:
                best_loss = loss
                mn_range = -st
                mx_range = st
            cnt += 1
            if cnt % 10 == 0:
                logger.info('{:.6f} loss at iter {}'.format(loss, cnt))
            st -= step
        return paddle.to_tensor(data=mn_range, dtype=self.dtype).to(self.device
            ), paddle.to_tensor(data=mx_range, dtype=self.dtype).to(self.device
            )

    def forward(self):
        best_range = self.search_migrate_range_1D()
        return self.get_best_scale(*best_range)

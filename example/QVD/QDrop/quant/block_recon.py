import sys
import paddle
import numpy as np
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, weight_get_quant_state
from .set_act_quantize_params import set_act_quantize_params


def block_reconstruction(model: QuantModel, block: BaseQuantBlock,
    cali_data: paddle.Tensor, batch_size: int=32, iters: int=20000, weight:
    float=0.01, opt_mode: str='mse', act_quant: bool=False, b_range: tuple=
    (20, 2), warmup: float=0.0, p: float=2.0, lr: float=4e-05, wwq: bool=
    True, waq: bool=True, order: str='together', input_prob: float=1.0,
    keep_gpu: bool=True):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    """
    """get input and set scale"""
    cached_inps, cached_outs = get_init(model, block, cali_data, wq=wwq, aq
        =waq, batch_size=batch_size, input_prob=True, keep_gpu=keep_gpu)
    if act_quant and order == 'together':
        set_act_quantize_params(block, cali_data=cached_inps[0][:min(256,
            cached_inps[0].shape[0])], awq=True, order=order)
    """set state"""
    cur_weight, cur_act = weight_get_quant_state(order, act_quant)
    block.set_quant_state(cur_weight, cur_act)
    """set quantizer"""
    round_mode = 'learned_hard_sigmoid'
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None
    for module in block.sublayers():
        """weight"""
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.
                weight_quantizer, round_mode=round_mode, weight_tensor=
                module.org_weight.data)
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
        """activation"""
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if (act_quant and order == 'together' and module.act_quantizer.
                delta is not None):
                module.act_quantizer.delta = (paddle.base.framework.
                    EagerParamBase.from_tensor(tensor=paddle.to_tensor(data
                    =module.act_quantizer.delta)))
                a_para += [module.act_quantizer.delta]
            """set up drop"""
            module.act_quantizer.is_training = True
    if len(w_para) != 0:
        w_opt = paddle.optimizer.Adam(parameters=w_para, weight_decay=0.0)
    if len(a_para) != 0:
        a_opt = paddle.optimizer.Adam(parameters=a_para, learning_rate=lr,
            weight_decay=0.0)
        tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=iters,
            eta_min=0.0, learning_rate=a_opt.get_lr())
        a_opt.set_lr_scheduler(tmp_lr)
        a_scheduler = tmp_lr
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight,
        max_count=iters, rec_loss=rec_loss, b_range=b_range, decay_start=0,
        warmup=warmup, p=p)
    device = 'cuda'
    sz = cached_inps[0].shape[0]
    for i in range(iters):
        idx = paddle.randint(low=0, high=sz, shape=(batch_size,))
        cur_inp, cur_sym = cached_inps[0][idx].to(device), cached_inps[1][idx
            ].to(device)
        if input_prob < 1.0:
            cur_inp = paddle.where(condition=paddle.rand(shape=cur_inp.
                shape, dtype=cur_inp.dtype) < input_prob, x=cur_inp, y=cur_sym)
        cur_out = cached_outs[idx].to(device)
        w_opt.clear_gradients(set_to_zero=False)
        if a_opt:
            a_opt.clear_gradients(set_to_zero=False)
        out_quant = block(cur_inp)
        err = loss_func(out_quant, cur_out)
        err.backward(retain_graph=True)
        w_opt.step()
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    paddle.device.cuda.empty_cache()
    for module in block.sublayers():
        if isinstance(module, QuantModule):
            """weight """
            module.weight_quantizer.soft_targets = False
        """activation"""
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.is_training = False
    """Case 3"""
    if act_quant and order == 'after' and waq:
        set_act_quantize_params(block, cached_inps[0], awq=True, order=order)


class LossFunction:

    def __init__(self, block: BaseQuantBlock, round_loss: str='relaxation',
        weight: float=1.0, rec_loss: str='mse', max_count: int=2000,
        b_range: tuple=(10, 2), decay_start: float=0.0, warmup: float=0.0,
        p: float=2.0):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup +
            (1 - warmup) * decay_start, start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(y=2) * grad.pow(y=2)).sum(axis=1
                ).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = paddle.sum(x=a * grad, axis=(1, 2, 3)).view(-1,
                1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'
                .format(self.rec_loss))
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_sublayers():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - 0.5).
                        abs() * 2).pow(y=b)).sum()
        else:
            raise NotImplementedError
        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print(
                'Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'
                .format(float(total_loss), float(rec_loss), float(
                round_loss), b, self.count))
        return total_loss


class LinearTempDecay:

    def __init__(self, t_max: int, rel_start_decay: float=0.2, start_b: int
        =10, end_b: int=2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, 1 -
                rel_t)

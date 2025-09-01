import paddle
import paddle.nn.functional as F
from models import vit,deit,swin
from models.swin import windows_partition, windows_reverse
from utils.calibrator import QuantCalibrator
from quantizers.adaround import AdaRoundQuantizer
from quant_layers import *
from types import MethodType
import logging
import random
import copy
import os

def patch_embed_forward(self, x):
    x = self.patch_embedding(x)
    x = x.flatten(2)  # [B, C, H, W] -> [B, C, h*w]
    x = x.transpose([0, 2, 1])  # [B, C, h*w] -> [B, h*w, C] = [B, N, C]
    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x


def vit_block_forward(self, x: paddle.Tensor) -> paddle.Tensor:
    h = x
    x = self.attn_norm(x)
    x = self.attn(x)
    #x = self.drop_path(x)
    x = x + h

    h = x
    x = self.mlp_norm(x)
    x = self.mlp(x)
    #x = self.drop_path(x)
    x = x + h

    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x

def deit_block_forward(self, x: paddle.Tensor) -> paddle.Tensor:
    h = x
    x = self.attn_norm(x)
    x = self.attn(x)
    x = self.drop_path(x)
    x = x + h

    h = x
    x = self.mlp_norm(x)
    x = self.mlp(x)
    x = self.drop_path(x)
    x = x + h
    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x

def swin_block_forward(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    h = x
    x = self.norm1(x)  # [batch, h*w, c]

    new_shape = [B, H, W, C]
    x = x.reshape(new_shape)  # [batch, h, w, c]

    if self.shift_size > 0:
        shifted_x = paddle.roll(x,
                                shifts=(-self.shift_size, -self.shift_size),
                                axis=(1, 2)) # [batch, h, w, c]
    else:
        shifted_x = x

    x_windows = windows_partition(shifted_x, self.window_size)  # [batch*n_windows, 7, 7, c]
    x_windows = x_windows.reshape(
        [-1, self.window_size * self.window_size, C])  # [batch*n_windows, 7*7, c]

    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [batch*n_windows, 7*7, c]
    attn_windows = attn_windows.reshape(
        [-1, self.window_size, self.window_size, C])  # [batch*n_windows, 7, 7, c]

    shifted_x = windows_reverse(attn_windows, self.window_size, H, W)   # [batch, h, w, c]

    # reverse cyclic shift
    if self.shift_size > 0:
        x = paddle.roll(shifted_x,
                        shifts=(self.shift_size, self.shift_size),
                        axis=(1, 2))
    else:
        x = shifted_x

    x = x.reshape([B, H*W, C])  # [batch, h*w, c]

    if self.drop_path is not None:
        x = h + self.drop_path(x)
    else:
        x = h + x

    h = x  # [batch, h*w, c]
    x = self.norm2(x)
    x = self.mlp(x)
    if self.drop_path is not None:
        x = h + self.drop_path(x)
    else:
        x = h + x
        
    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x


def swin_patch_embed_forward(self, x):
    x = self.patch_embed(x)  # [batch, embed_dim, h, w]; h,w = patch_resolution
    x = x.flatten(start_axis=2, stop_axis=-1)  # [batch, embed_dim, h*w]; h*w = num_patches
    x = x.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
    x = self.norm(x)  # [batch, num_patches, embed_dim]
    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x

def swin_patch_merging_forward(self, x):
    h, w = self.input_resolution
    b, _, c = x.shape
    x = x.reshape([b, h, w, c])

    x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
    x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
    x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
    x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
    x = paddle.concat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
    x = x.reshape([b, -1, 4*c])  # [B, H/2*W/2, 4*C]

    x = self.norm(x)
    x = self.reduction(x)
    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x

class BlockReconstructor(QuantCalibrator):
    def __init__(self, model, full_model, calib_loader, metric="hessian_perturb", use_mean_hessian=True, temp=20):
        super().__init__(model, calib_loader)
        self.full_model = full_model
        self.metric = metric
        self.use_mean_hessian = use_mean_hessian
        self.blocks = {}
        self.full_blocks = {}
        self.quanted_blocks = []
        self.raw_pred_softmaxs = None
        self.temperature = temp
        self.calib_data=[]
        types_of_block = [
            vit.PatchEmbedding,
            vit.TransformerLayer,
            deit.PatchEmbedding,
            deit.TransformerLayer,
            swin.PatchEmbedding,
            swin.SwinTransformerBlock,
            swin.PatchMerging,
        ]
        b=True
        for name, module in self.model.named_sublayers(include_self=True):
            if any(isinstance(module, t) for t in types_of_block) or name.split('.')[-1] == 'classifier':
                if name.split('.')[-1] == '0':
                    b=True
                if b:
                    self.blocks[name] = module
                    BlockReconstructor._prepare_module_data_init(module)
        b=True          
        for name_, module_ in self.full_model.named_sublayers(include_self=True):
            if any(isinstance(module_, t) for t in types_of_block) or name_.split('.')[-1] == 'classifier':
                if name_.split('.')[-1] == '0':
                    b=True
                if b:
                    self.full_blocks[name_] = module_
                    BlockReconstructor._prepare_module_data_init(module_)
                
    @staticmethod
    def _prepare_module_data_init(module):
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
        module.raw_grad = module.tmp_grad = None
        module.quanted_input = None
        if isinstance(module, vit.PatchEmbedding) or isinstance(module, deit.PatchEmbedding):
            module.forward = MethodType(patch_embed_forward, module)
        elif isinstance(module, vit.TransformerLayer):
            module.forward = MethodType(vit_block_forward, module)
        elif isinstance(module, deit.TransformerLayer):
            module.forward = MethodType(deit_block_forward, module)
        elif isinstance(module, swin.PatchEmbedding):
            module.forward = MethodType(swin_patch_embed_forward, module)
        elif isinstance(module, swin.PatchMerging):
            module.forward = MethodType(swin_patch_merging_forward, module)
        elif isinstance(module, swin.SwinTransformerBlock):
            module.forward = MethodType(swin_block_forward, module)
        module.perturb_u = module.perturb_d = False
                
    def set_block_mode(self, block, qmode='raw'):
        for _, module in block.named_sublayers(include_self=True):
            if hasattr(module, 'qmode'):
                module.qmode = qmode

    def replace_block(self, target_block, new_block):
        self._replace_block_recursive(self.model, target_block, new_block)

    def _replace_block_recursive(self, model, target_block, new_block):
        for name, child in model.named_children():
            if child is target_block:
                setattr(model, name, new_block)
            else:
                self._replace_block_recursive(child, target_block, new_block)
                
    def wrap_quantizers_in_net(self, block, name):
        print('wraping quantizers in {} ...'.format(name))
        for name, module in block.named_sublayers(include_self=True):
            if hasattr(module, 'w_quantizer'):
                if isinstance(module, MinMaxQuantLinear):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.reshape([module.in_features,module.n_V, module.crb_rows]), 
                                                           round_mode='learned_hard_sigmoid')
                elif isinstance(module, MinMaxQuantConv2d):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.reshape([module.weight.shape[0], -1]), 
                                                           round_mode='learned_hard_sigmoid')
                module.w_quantizer.soft_targets = True

    def set_qdrop(self, block, prob):
        for _, module in block.named_sublayers(include_self=True):
            if hasattr(module, 'qmode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    if hasattr(module.a_quantizer, 'drop_prob'):
                        module.a_quantizer.drop_prob = prob
                elif isinstance(module, MinMaxQuantMatMul):
                    if hasattr(module.A_quantizer, 'drop_prob'):
                        module.A_quantizer.drop_prob = prob
                    if hasattr(module.B_quantizer, 'drop_prob'):
                        module.B_quantizer.drop_prob = prob

    def init_block_raw_data(self, block, full_block, name, qinp=False):
        self.init_block_raw_inp_outp(block, full_block, name)
        if qinp and 'patch_embed' not in name:
            self.init_block_quanted_input(block, full_block, name)
        
        if self.metric == "hessian_perturb":
            self.init_block_perturb_hessian(block, full_block, name)
        elif self.metric == "hessian_brecq":
            self.init_block_brecq_hessian(block, full_block, name)

        if 'patch_embed' in name:
            block.quanted_input = block.raw_input

    def init_block_raw_inp_outp(self, block, full_block, name):
        logging.info('initializing raw input and raw output ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        hooks = []
        hooks.append(full_block.register_forward_post_hook(self.outp_forward_hook))
        hooks.append(full_block.register_forward_post_hook(self.single_input_forward_hook))
        need_calculate_raw_softmax = False
        if self.raw_pred_softmaxs is None and self.metric in ["hessian_brecq", "hessian_perturb"]:
            need_calculate_raw_softmax = True
            self.raw_pred_softmaxs = []
        with paddle.no_grad():
            self.calib_data=[]
            for inp, target in self.calib_loader:
                self.calib_data.append(inp)
                pred = self.full_model(inp) / self.temperature
                if need_calculate_raw_softmax:
                    raw_pred_softmax = F.softmax(pred, axis=-1).detach()
                    self.raw_pred_softmaxs.append(raw_pred_softmax)
            paddle.device.cuda.empty_cache()
        block.raw_out = paddle.concat(full_block.tmp_out, axis=0)
        block.raw_input = paddle.concat(full_block.tmp_input, axis=0)
        full_block.tmp_input, full_block.tmp_out = None, None
        for hook in hooks:
            hook.remove()

    def init_block_quanted_input(self, block, full_block, name):
        logging.info('initializing quanted input ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        self.replace_block(block, full_block)
        hook = full_block.register_forward_post_hook(self.single_input_forward_hook)
        with paddle.no_grad():
            for i, inp in enumerate(self.calib_data):
                pred = self.model(inp)
        paddle.device.cuda.empty_cache()
        block.quanted_input = paddle.concat(full_block.tmp_input, axis=0)
        full_block.tmp_input = None
        hook.remove()
        self.replace_block(full_block, block)
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')

    def init_block_perturb_hessian(self, block, full_block, name):
        logging.info('initializing perturbation hessian ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        raw_grads = []
        for step in range(2):
            full_block.hooks=[]
            hook = full_block.register_forward_post_hook(self.outp_forward_hook_for_grad)
            full_block.perturb_u, full_block.perturb_d = (step == 0, step == 1)
            for i, inp in enumerate(self.calib_data):
                self.model.clear_gradients()
                pred = self.full_model(inp) / self.temperature
                loss = F.kl_div(F.log_softmax(pred, axis=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
                loss.backward()
            paddle.device.cuda.empty_cache()
            raw_grads.append(paddle.concat(full_block.tmp_grad, axis=0))
            full_block.tmp_grad = None
            full_block.perturb_u = full_block.perturb_d = False
            hook.remove()
            for hook_ in full_block.hooks:
                hook_.remove()
        block.raw_grad = (raw_grads[0] - raw_grads[1]).abs()
        block.raw_grad = block.raw_grad.mean(axis=0, keepdim=True) if self.use_mean_hessian else block.raw_grad
        
    def init_block_brecq_hessian(self, block, full_block, name):
        logging.info('initializing brecq hessian ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        self.replace_block(block, full_block)
        full_block.hooks=[]
        hook = full_block.register_forward_post_hook(self.outp_forward_hook_for_grad)
        for i, inp in enumerate(self.calib_data):
            self.model.clear_gradients()
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, axis=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
            loss.backward()
        paddle.device.cuda.empty_cache()
        raw_grads = paddle.concat(full_block.tmp_grad, axis=0)
        full_block.tmp_grad = None
        block.raw_grad = raw_grads.abs().pow(2)
        hook.remove()
        for hook_ in full_block.hooks:
            hook_.remove()
        self.replace_block(full_block, block)
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
            
    def reconstruct_single_block(self, name, block,
                                 batch_size: int = 32, iters: int = 20000, weight: float = 0.01,
                                 b_range: tuple = (20, 2), warmup: float = 0.2, lr: float = 4e-5, p: float = 2.0, 
                                 quant_act = False, mode = 'qdrop', drop_prob: float = 1.0):
        self.wrap_quantizers_in_net(block, name)
        self.set_block_mode(block, 'quant_forward')
        for _name, module in block.named_sublayers(include_self=True):
            if hasattr(module, 'training_mode'):
                module.init_training()
        if mode == 'qdrop':
            self.set_qdrop(block, drop_prob)
        w_params, a_params = [], []
        for _name, module in block.named_sublayers(include_self=True):
            if hasattr(module, 'qmode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    w_params += [module.w_quantizer.alpha]
                    if quant_act:
                        module.a_quantizer.scale.requires_grad = True
                        a_params += [module.a_quantizer.scale]
                    else:
                        module.qmode = 'debug_only_quant_weight'
                elif isinstance(module, MinMaxQuantMatMul):
                    if quant_act:
                        module.A_quantizer.scale.requires_grad = True
                        module.B_quantizer.scale.requires_grad = True
                        a_params += [module.A_quantizer.scale, module.B_quantizer.scale]
                    else:
                        module.qmode = 'raw'
        w_optimizer = paddle.optimizer.Adam(parameters=w_params)
        a_optimizer = paddle.optimizer.Adam(parameters=a_params, learning_rate=lr) if len(a_params) != 0 else None
        a_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=a_optimizer.get_lr(), T_max=iters, eta_min=0.) if len(a_params) != 0 else None
        loss_func = LossFunction(block, round_loss='relaxation', weight=weight, max_count=iters, 
                                 rec_loss=self.metric if 'classifier' not in name else 'kl_div',
                                 b_range=b_range, decay_start=0, warmup=warmup, p=p)

        for it in range(iters):
            idx = paddle.randperm(block.raw_input.shape[0])[:batch_size]
            if mode == 'qdrop':
                cur_quant_inp = block.quanted_input[idx] if block.quanted_input is not None else block.raw_input[idx]
                cur_fp_inp = block.raw_input[idx]
                cur_inp = paddle.where(paddle.rand(shape=cur_quant_inp.shape, dtype=cur_quant_inp.dtype) < drop_prob, cur_quant_inp, cur_fp_inp)
            elif mode == 'rinp':
                cur_inp = block.raw_input[idx]
            elif mode == 'qinp':
                cur_inp = block.quanted_input[idx]
            cur_out = block.raw_out[idx].detach()
            if block.raw_grad is not None:
                if self.metric == "hessian_perturb" and self.use_mean_hessian:
                    cur_grad = block.raw_grad.detach()
                elif self.metric == "hessian_brecq" or not self.use_mean_hessian:
                    cur_grad = block.raw_grad[idx].detach()
                else:
                    cur_grad = None
            else:
                cur_grad = None
            w_optimizer.clear_grad()
            if quant_act:
                a_optimizer.clear_grad()
            out_quant = block(cur_inp.detach())
            if 'classifier' not in name:
                err = loss_func(out_quant, cur_out, cur_grad)
            else:
                err = loss_func(out_quant, cur_out)
            err.backward(retain_graph=True)
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()
        w_optimizer.clear_grad()
        if quant_act:
            a_optimizer.clear_grad()
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input
        paddle.device.cuda.empty_cache()
        # Finish optimization, use hard rounding.
        for name, module in block.named_sublayers(include_self=True):
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
                module.weight.data= paddle.assign(module.w_quantizer.get_hard_value(module.weight.data))
                del module.w_quantizer.alpha
                module.w_quantizer.round_mode = "nearest"
            if hasattr(module, 'qmode'):
                module.qmode = 'raw'
            if hasattr(module, 'training_mode'):
                module.end_training()
        self.set_qdrop(block, 1.0)
        
        paddle.device.cuda.empty_cache()

    def reconstruct_model(self, quant_act: bool = False, mode: str = 'qdrop+', drop_prob: float = 1.0, keep_gpu: bool = True,root_path=None):
        for name, module in self.model.named_sublayers(include_self=True):
            if hasattr(module, 'qmode'):
                module.qmode = 'raw'
        for idx, name in enumerate(self.blocks.keys()):
            block, full_block = self.blocks[name], self.full_blocks[name]
            logging.info('reconstructing {} ...'.format(name))
            self.init_block_raw_data(block, full_block, name, qinp=(mode != 'rinp'))
            logging.info('adaround training for {} ...'.format(name))
            self.reconstruct_single_block(name, block, quant_act=quant_act, mode=mode, drop_prob=drop_prob)
            self.quanted_blocks.append(name)
            logging.info('finished reconstructing {}.'.format(name))
            save_path = os.path.join(root_path, 'optimized_{}.pth'.format(name))
            state_dict = dict()
            state_dict['model'] = self.model.state_dict()
            paddle.save(state_dict, save_path)
        for name, module in self.model.named_sublayers(include_self=True):
            if hasattr(module, 'qmode'):
                module.qmode = 'quant_forward'
            # if hasattr(module, 'w_quantizer'):
            #     module.weight.data.copy_(module.w_quantizer.get_hard_value(module.weight.data))
            #     del module.w_quantizer.alpha
            #     module.w_quantizer.round_mode = "nearest"

        
class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.init_loss = 0
        self.count = 0
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def __call__(self, pred, tgt, grad=None, count=True):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute Hessian
        :return: total loss function
        """
        if count:
            self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0)
        elif self.rec_loss in ['hessian_brecq', 'hessian_perturb']:
            rec_loss = ((pred - tgt).pow(2) * grad.abs()).mean()
            
        elif self.rec_loss == 'kl_div':
            rec_loss = F.kl_div(F.log_softmax(pred, axis=-1), F.softmax(tgt, axis=-1), reduction="batchmean")
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_sublayers(include_self=True):
                if hasattr(module, 'w_quantizer'):
                    round_vals = module.w_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError
        if self.count == 1:
            self.init_loss = rec_loss.item()
        rec_loss = rec_loss * 2 / self.init_loss
        total_loss = rec_loss + round_loss
        if self.count == 1 or self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
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
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

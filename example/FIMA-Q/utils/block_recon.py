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
import gc
def clear_memory():
    """全面内存清理函数"""
    # 1. 清除CUDA缓存
    paddle.device.cuda.empty_cache()
    
    # 2. 垃圾回收
    
    gc.collect()
    
    # 3. 显式清除中间变量
    for var in list(locals().keys()):
        if isinstance(locals()[var], paddle.Tensor):
            locals()[var]._clear()
    
    # 4. 清除全局变量引用
    for name in dir():
        if not name.startswith('_') and isinstance(globals()[name], paddle.Tensor):
            globals()[name] = None
def patch_embed_forward(self, x):
    x = self.patch_embedding(x)
    x = x.flatten(2)  # [B, C, H, W] -> [B, C, h*w]
    x = x.transpose([0, 2, 1])  # [B, C, h*w] -> [B, h*w, C] = [B, N, C]
    if self.perturb:
        rand_perturb = paddle.empty_like(x, dtype=paddle.float).uniform(1, 2) * self.r
        x = x + rand_perturb
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

    if self.perturb:
        rand_perturb = paddle.empty_like(x, dtype=paddle.float).uniform(1, 2) * self.r
        x = x + rand_perturb
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
    if self.perturb:
        rand_perturb = paddle.empty_like(x, dtype=paddle.float).uniform(1, 2) * self.r
        x = x + rand_perturb
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
        
    if self.perturb:
        rand_perturb = paddle.empty_like(x, dtype=paddle.float).uniform(1, 2) * self.r
        x = x + rand_perturb
    return x


def swin_patch_embed_forward(self, x):
    x = self.patch_embed(x)  # [batch, embed_dim, h, w]; h,w = patch_resolution
    x = x.flatten(start_axis=2, stop_axis=-1)  # [batch, embed_dim, h*w]; h*w = num_patches
    x = x.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
    x = self.norm(x)  # [batch, num_patches, embed_dim]
    if self.perturb:
        rand_perturb = paddle.empty_like(x, dtype=paddle.float).uniform(1, 2) * self.r
        x = x + rand_perturb
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
    if self.perturb:
        rand_perturb = paddle.empty_like(x, dtype=paddle.float).uniform(1, 2) * self.r
        x = x + rand_perturb
    return x

class BlockReconstructor(QuantCalibrator):
    def __init__(self, model, full_model,calib_loader, metric="mse", use_mean_hessian=True, temp=20,k=1,dis_mode='q',p1=2.,p2=2.):
        super().__init__(model, calib_loader)
        self.full_model = full_model
        self.metric = metric
        self.use_mean_hessian = use_mean_hessian
        self.k=k
        self.dis_mode=dis_mode
        self.p1=p1
        self.p2=p2
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
                if name.split('.')[-1] == 'classifier':
                    b=True
                if b:
                    self.blocks[name] = module
                    BlockReconstructor._prepare_module_data_init(module)
        b=True          
        for name_, module_ in self.full_model.named_sublayers(include_self=True):
            if any(isinstance(module_, t) for t in types_of_block) or name_.split('.')[-1] == 'classifier':
                if name_.split('.')[-1] == 'classifier':
                    b=True
                if b:
                    self.full_blocks[name_] = module_
                    BlockReconstructor._prepare_module_data_init(module_)
                
    @staticmethod
    def _prepare_module_data_init(module):
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
        module.raw_grad = module.tmp_grad = None
        module.quanted_input = module.quanted_out = None
        module.delta_out = module.inverse_B = None
        module.r=1e-6
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
        module.perturb = False

                
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
        
        # if self.metric == "fisher_brecq":
        #     self.init_block_brecq_hessian(block, full_block, name)

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
        if self.raw_pred_softmaxs is None and self.metric in ["fisher_brecq", "fisher_lr","fisher_diag","fisher_dplr"]:
            need_calculate_raw_softmax = True
            self.raw_pred_softmaxs = []
        with paddle.no_grad():
            self.calib_data=[]
            for inp, target in self.calib_loader:
                self.calib_data.append(inp)
                pred = self.full_model(inp) / self.temperature
                if need_calculate_raw_softmax:
                    raw_pred_softmax = F.softmax(pred, axis=-1).cpu().detach()
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
        #self.replace_block(block, full_block)
        hook = block.register_forward_post_hook(self.single_input_forward_hook)
        with paddle.no_grad():
            for i, inp in enumerate(self.calib_data):
                pred = self.model(inp)
        paddle.device.cuda.empty_cache()
        block.quanted_input = paddle.concat(block.tmp_input, axis=0)
        block.tmp_input = None
        hook.remove()
        #self.replace_block(full_block, block)
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')

    def new_fisher_ro(self, block,op1,op2):
        logging.info('updating fisher information matrix ...')
        hooks = []
        block.hooks=[]
        hooks.append(block.register_forward_post_hook(self.outp_forward_hook))
        hooks.append(block.register_forward_post_hook(self.outp_forward_hook_for_grad))
        for i, inp in enumerate(self.calib_data):
            op1.clear_grad()
            op2.clear_grad()
            self.model.clear_gradients()
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, axis=-1), self.raw_pred_softmaxs[i])
            loss.backward(retain_graph=True) 
            del loss,pred,inp
            clear_memory()
        raw_grad = paddle.concat(block.tmp_grad, axis=0)
        raw_grad = raw_grad.reshape([raw_grad.shape[0], -1]).abs()
        raw_grad = raw_grad.mean(axis=0).unsqueeze(0) # (1, N)
        q_out = paddle.concat(block.tmp_out, axis=0)
        delta_out = (q_out - block.raw_out).abs().mean(axis=0).reshape([1, -1]) # (1, N)
        block.tmp_grad = block.tmp_out = None
        for hook in hooks:
            hook.remove()
        for hook_ in block.hooks:
            hook_.remove()
        if block.raw_grad is None:
            block.raw_grad = raw_grad
            block.delta_out = delta_out
        else:
            block.raw_grad = paddle.concat([block.raw_grad, raw_grad], axis=0) # (k, N)
            block.delta_out = paddle.concat([block.delta_out, delta_out], axis=0) # (k, N)
        block.inverse_B = paddle.linalg.inv(block.delta_out @ block.delta_out.transpose([1, 0])) # (k, k)
        # block.inverse_B = torch.eye(block.raw_grad.shape[0]).to(device)
        del raw_grad, delta_out
        paddle.device.cuda.empty_cache()
   
            
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
        i_change=math.floor(iters/self.k)
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

            loss_func.update_fisher = False
            if loss_func.rec_loss in ["fisher_lr", "fisher_diag", "fisher_dplr"] :
                if self.dis_mode in ['q']:
                    if it % i_change == 0:
                        self.new_fisher_ro(block,w_optimizer,a_optimizer)
                        loss_func.update_fisher = True
                elif self.dis_mode in ['qf']:
                    if it in range(self.k):
                        self.new_fisher_ro(block,w_optimizer,a_optimizer)
                        loss_func.update_fisher = True
                cur_grad = block.raw_grad if block.raw_grad is not None else None
            elif self.metric == "fisher_brecq" :
                cur_grad = block.raw_grad[idx]
            else:
                cur_grad = None

            w_optimizer.clear_grad()
            if quant_act:
                a_optimizer.clear_grad()
            out_quant = block(cur_inp)
            if 'classifier' not in name:
                err = loss_func(out_quant, cur_out, cur_grad)
            else:
                err = loss_func(out_quant, cur_out)
            err.backward(retain_graph=True)
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()
            paddle.device.cuda.empty_cache()
        w_optimizer.clear_grad()
        if quant_act:
            a_optimizer.clear_grad()
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input,block.delta_out,block.inverse_B,loss_func
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
                 p: float = 2.,
                 p1: float = 2.,
                 p2: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.p1 = p1
        self.p2 = p2
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.update_fisher = False
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

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
            rec_loss = self.lp_loss(pred, tgt, p=self.p) / 10
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0) / 10
        elif self.rec_loss == 'fisher_lr':
            cha = (pred - tgt).abs().reshape([pred.shape[0], -1])
            loss_1 = (cha * grad.abs()).mean(axis=-1).pow(2).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_1 = loss_1.detach()
            rec_loss = 2 * loss_1 / self.init_loss_1
        elif self.rec_loss == 'fisher_diag':
            cha = (pred - tgt).abs().reshape([pred.shape[0], -1])
            loss_2 = (cha.pow(2) * grad.abs().mean(axis=0)).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_2 = loss_2.detach()
            rec_loss = 2 * loss_2 / self.init_loss_2
        elif self.rec_loss == 'fisher_dplr':
            cha = (pred - tgt).abs().reshape([pred.shape[0], -1])
            A = cha.unsqueeze(1) @ grad.abs().transpose([1, 0])
            loss_1 = (A @ self.block.inverse_B @ A.transpose([0,2, 1])).mean()
            loss_2 = (cha.pow(2) * grad.abs().mean(axis=0)).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_1 = loss_1.detach()
                self.init_loss_2 = loss_2.detach()
            rec_loss = self.p1 * loss_1 / self.init_loss_1 + self.p2 * loss_2 / self.init_loss_2
        elif self.rec_loss == 'fisher_brecq':
            loss_1 = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
            if self.count == 1:
                self.init_loss_1 = loss_1.detach()
            rec_loss = loss_1 / self.init_loss_1
        elif self.rec_loss == 'kl_div':
            rec_loss = F.kl_div(F.log_softmax(pred, axis=-1), F.softmax(tgt, axis=-1).detach(), reduction="batchmean")
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = round_loss_pow2 = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_sublayers(include_self=True):
                if hasattr(module, 'w_quantizer'):
                    round_vals = module.w_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

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

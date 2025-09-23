
import paddle
import paddle.nn.functional as F
from utils.calibrator import QuantCalibrator
from types import MethodType
import logging
import os

def mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    if self.perturb_u:
        x = x + paddle.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - paddle.ones_like(x) * 1e-6
    return x


def positive_percentile(tensor, pct):
    logging.info('positive_percentile ...')
    mini_batch_size = 8
    tensor_too_large = True
    while tensor_too_large:
        print(mini_batch_size)
        try:
            t = tensor.reshape([mini_batch_size, -1])[0:1, :]
            t = t.reshape([-1])
            positive_mask = t > 0
            positive_tensor = paddle.where(positive_mask, t, paddle.to_tensor(float('nan')))
            sorted_tensor = positive_tensor.sort(axis=0)
            tensor_too_large = False
        except:
            mini_batch_size *= 2
    counts = (~paddle.isnan(sorted_tensor)).sum(axis=0, keepdim=True).astype('float32')
    ranks = ((counts * pct).ceil().astype('int64') - 1).clip(min=0)
    result = paddle.gather(sorted_tensor, index=ranks,axis=0).squeeze()
    return result.item()


class MLPReconstructor(QuantCalibrator):
    def __init__(self, model, full_model, calib_loader, metric="hessian_perturb", use_mean_hessian=True, temp=20):
        super().__init__(model, calib_loader)
        self.full_model = full_model
        self.metric = metric
        self.use_mean_hessian = use_mean_hessian
        self.blocks = {}
        self.full_blocks = {}
        self.raw_pred_softmaxs = None
        self.temperature = temp
        self.calib_data = []

        now='1'
        start=False
        for name, module in self.model.named_sublayers(include_self=True):
            if name.split('.')[-1]==now :start=True
            if len(name.split('.')) >= 2 and (name.split('.')[-2] == 'layers' or name.split('.')[-2] == 'blocks') and start:
                #if int(name.split('.')[-1]) < now:continue
                self.blocks[name] = module
                MLPReconstructor._prepare_module_data_init(module)
        start=False
        for name_, module_ in self.full_model.named_sublayers(include_self=True) :
            if name_.split('.')[-1]==now :start=True
            if len(name_.split('.')) >= 2 and (name_.split('.')[-2] == 'layers' or name_.split('.')[-2] == 'blocks') and start:
                #if int(name_.split('.')[-1]) < now:continue
                self.full_blocks[name_] = module_
                MLPReconstructor._prepare_module_data_init(module_)

    @staticmethod
    def _prepare_module_data_init(module):
        module.mlp.fc1.raw_input = module.mlp.fc1.tmp_input = None
        module.mlp.fc2.raw_input = module.mlp.fc2.tmp_input = None
        module.mlp.raw_out = module.mlp.tmp_out = None
        module.mlp.raw_grad = module.mlp.tmp_grad = None
        module.mlp.forward = MethodType(mlp_forward, module.mlp)
        module.mlp.perturb_u = module.mlp.perturb_d = False

    def init_block_raw_data(self, block):
        self.init_block_raw_inp_outp(block)
        if self.metric == "hessian_perturb":
            self.init_block_perturb_hessian(block)
        paddle.device.cuda.empty_cache()

    def init_block_raw_inp_outp(self, block):
        logging.info('init_block_raw_inp_outp ...')
        hooks = []
        hooks.append(block.mlp.register_forward_post_hook(self.outp_forward_hook))
        hooks.append(block.mlp.fc1.register_forward_post_hook(self.single_input_forward_hook))
        hooks.append(block.mlp.fc2.register_forward_post_hook(self.single_input_forward_hook))
        need_calculate_raw_softmax = False
        if self.raw_pred_softmaxs is None and self.metric == "hessian_perturb":
            need_calculate_raw_softmax = True
            self.raw_pred_softmaxs = []
        with paddle.no_grad():
            self.calib_data=[]
            for inp, target in self.calib_loader:
                self.calib_data.append(inp)
                pred = self.full_model(inp) / self.temperature
                if need_calculate_raw_softmax:
                    raw_pred_softmax = F.softmax(pred, axis=-1)
                    self.raw_pred_softmaxs.append(raw_pred_softmax)
            
        block.mlp.raw_out = paddle.concat(block.mlp.tmp_out, axis=0)
        block.mlp.fc1.raw_input = paddle.concat(block.mlp.fc1.tmp_input, axis=0)
        block.mlp.fc2.raw_input = paddle.concat(block.mlp.fc2.tmp_input, axis=0)
        block.mlp.fc1.tmp_input = block.mlp.fc2.tmp_input = block.mlp.tmp_out = None
        for hook in hooks:
            hook.remove()
        paddle.device.cuda.empty_cache()

    def init_block_perturb_hessian(self, block):
        logging.info('init_block_perturb_hessian ...')
        raw_grads = []
        for step in range(2):
            block.mlp.hooks = []
            hook = block.mlp.register_forward_post_hook(self.outp_forward_hook_for_grad)
            block.mlp.perturb_u, block.mlp.perturb_d = (step == 0, step == 1)
            for i,inp in enumerate(self.calib_data):
                self.model.clear_gradients()
                pred = self.full_model(inp) / self.temperature
                loss = F.kl_div(F.log_softmax(pred, axis=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
                loss.backward()
            paddle.device.cuda.empty_cache()
            raw_grads.append(paddle.concat(block.mlp.tmp_grad, axis=0))
            block.mlp.tmp_grad = None
            block.mlp.perturb_u = block.mlp.perturb_d = False
            hook.remove()
            for hook_ in block.mlp.hooks:
                hook_.remove()
        block.mlp.raw_grad = (raw_grads[0] - raw_grads[1]).abs()
        block.mlp.raw_grad = block.mlp.raw_grad.mean(axis=0, keepdim=True) if self.use_mean_hessian else block.mlp.raw_grad
        block.mlp.raw_grad = block.mlp.raw_grad * paddle.sqrt(block.mlp.raw_grad.size / block.mlp.raw_grad.pow(2).sum())
            
    def reconstruct_single_block(self, name, block, ub,
                                 batch_size: int = 32, iters: int = 20000, lr: float = 4e-5, p: float = 2.0):
        w_params = []
        logging.info('reconstruct_single_block {} ...'.format(name))
        for _name, module in block.named_sublayers(include_self=True):
            if 'fc1' in _name or 'fc2' in _name :
                w_params += [module.weight, module.bias]
        w_optimizer = paddle.optimizer.Adam(parameters=w_params, learning_rate=lr)
        w_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=w_optimizer.get_lr(), T_max=iters, eta_min=0.)
        loss_func = LossFunction(block, weight=2.0, rec_loss=self.metric, max_count=iters, p=p)
        for i in range(iters):
            idx = paddle.randperm(block.mlp.fc1.raw_input.shape[0])[:batch_size]
            cur_inp = block.mlp.fc1.raw_input[idx]
            cur_out = block.mlp.raw_out[idx]
            if self.metric == "hessian_perturb":
                cur_grad = block.mlp.raw_grad if self.use_mean_hessian else block.mlp.raw_grad[idx]
            else:
                cur_grad = None
            
            w_optimizer.clear_grad()
            recon_out = block.mlp(cur_inp)
            fc2_inp = block.mlp.act(block.mlp.fc1(cur_inp))
            fc2_quant_inp = paddle.clip(fc2_inp, 0, ub)
            quant_out = block.mlp.fc2(fc2_quant_inp)
            err = loss_func(recon_out, cur_out, cur_grad, quant_out)
            err.backward()
            w_optimizer.step()
            w_scheduler.step()
        del block.mlp.fc1.raw_input, block.mlp.raw_out, block.mlp.raw_grad
        paddle.device.cuda.empty_cache()

    def reconstruct_model(self, pct,model_name,root_path):
        for name, block in self.blocks.items():
            logging.info('reconstructing {} ...'.format(name))
            full_block = self.full_blocks[name]
            self.init_block_raw_data(full_block)
            block.mlp.fc1.raw_input = full_block.mlp.fc1.raw_input
            block.mlp.raw_out = full_block.mlp.raw_out
            if self.metric == "hessian_perturb":
                block.mlp.raw_grad = full_block.mlp.raw_grad
                del full_block.mlp.raw_grad
            ub = positive_percentile(full_block.mlp.fc2.raw_input, pct=pct)
            del full_block.mlp.fc1.raw_input, full_block.mlp.fc2.raw_input, full_block.mlp.raw_out
            paddle.device.cuda.empty_cache()
            logging.info('ub: {}'.format(ub))
            self.reconstruct_single_block(name, block, ub=ub)
            logging.info('finished reconstructing {}.'.format(name))
            save_path = os.path.join(root_path, '{}_reconstructed_{}.pth'.format(model_name,name))
            state_dict = dict()
            state_dict['model'] = self.model.state_dict()
            paddle.save(state_dict, save_path)

        
class LossFunction:
    def __init__(self,
                 block,
                 weight: float = 2.0,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.2,
                 p: float = 2.):
        self.block = block
        self.rec_loss = rec_loss
        self.weight = weight
        self.p = p
        self.count = 0
        self.loss_start = max_count * warmup
        self.p = p
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def __call__(self, pred, tgt, grad=None, quant_out=None):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=self.p) / 10
            quant_loss = self.lp_loss(quant_out, tgt, p=self.p) / 10
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0) / 10
        elif self.rec_loss == 'hessian_perturb':
            rec_loss = ((pred - tgt).pow(2) * grad.abs()).sum(1).mean() / 10
            quant_loss = ((quant_out - tgt).pow(2) * grad.abs()).sum(1).mean() / 10
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        total_loss = rec_loss + quant_loss * self.weight
        if self.count == 1 or self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, quant:{:.3f})\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(quant_loss), self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
            
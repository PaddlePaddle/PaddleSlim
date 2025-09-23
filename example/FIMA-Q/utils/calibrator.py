import numpy as np
import paddle
import paddle.nn.functional as F
from tqdm import tqdm
from quant_layers import MinMaxQuantMatMul, MinMaxQuantConv2d, MinMaxQuantLinear
import functools

class QuantCalibrator:
    def __init__(self, model, calib_loader):
        self.model = model
        self.calib_loader = calib_loader
        
    def single_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append(inp[0].cpu().detach())
        
    def double_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1].cpu().detach())
    
    def outp_forward_hook(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append(outp.clone().cpu().detach())

    def outp_forward_hook_for_grad(self, module, inp, outp):
        module.hooks.append(outp.register_hook(functools.partial(self.grad_hook, module=module)))
        
    def grad_hook(self, grad,module):
        if module.tmp_grad is None:
            module.tmp_grad = []
        module.tmp_grad.append(grad.clone().cpu().detach())

    def batching_quant_calib(self):
        raw_pred_softmaxs = []
        self.calib_data=[]
        with paddle.no_grad():
            for inp, target in self.calib_loader:
                self.calib_data.append(inp)
                pred = self.model(inp)
                raw_pred_softmax = F.softmax(pred, axis=-1).detach()
                raw_pred_softmaxs.append(raw_pred_softmax)
            paddle.device.cuda.empty_cache()

        total = sum(1 for name, module in self.model.named_sublayers(include_self=True) if hasattr(module, 'metric') and not module.calibrated)
        with tqdm(total=total) as progress_bar:
            for name, module in self.model.named_sublayers(include_self=True):
                if not hasattr(module, 'metric') or module.calibrated:
                    continue
                progress_bar.set_description(f"calibrating {name}")
                hooks = []
                hooks.append(module.register_forward_post_hook(self.outp_forward_hook))
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    hooks.append(module.register_forward_post_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantMatMul):
                    hooks.append(module.register_forward_post_hook(self.double_input_forward_hook))
                for i, inp in enumerate(self.calib_data):
                    self.model.clear_gradients()
                    pred = self.model(inp)
                paddle.device.cuda.empty_cache()
                # replace cached raw_inputs, raw_outs
                module.raw_out = paddle.concat(module.tmp_out, axis=0)
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    module.raw_input = paddle.concat(module.tmp_input, axis=0)
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [paddle.concat(_, axis=0) for _ in module.tmp_input]
                for hook in hooks:
                    hook.remove()
                module.tmp_input = module.tmp_out = None
                # run hyperparameter_searching
                with paddle.no_grad():
                    module.hyperparameter_searching()
                    if hasattr(module, 'prev_layer') and module.prev_layer is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam()
                    paddle.device.cuda.empty_cache()
                # if isinstance(module, MinMaxQuantLinear):
                #     print(name,"scale:",module.a_quantizer.scale,module.w_quantizer.scale)
                #     print(name,"zp:",module.a_quantizer.zero_point,module.w_quantizer.zero_point)
                progress_bar.update()
        # end calibration
        for name, module in self.model.named_sublayers(include_self=True):
            if hasattr(module, 'qmode'):
                module.qmode = "quant_forward"

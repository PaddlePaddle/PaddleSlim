import paddle
from numpy import isin
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, SoSPTQSLBatchingQuantMatMul
from tqdm import tqdm
# from quant_layers.fq_vit import QIntSoftmax, QIntLayerNorm, QAct, QConv2d, QLinear
import paddle.nn.functional as F


class QuantCalibrator:
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume 
    hundreds of GB of memory.
    """

    def __init__(self, net, wrapped_modules, calib_loader, sequential=True):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
        self.blockwise_modules = {}

    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        n_calibration_steps = 2
        for step in range(n_calibration_steps):
            print(f'Start calibration step={step + 1}')
            for name, module in self.wrapped_modules.items():
                if hasattr(module, 'calibrated'):
                    if step == 1:
                        module.mode = 'raw'
                    elif step == 2:
                        module.mode = 'quant_forward'
                else:
                    module.mode = f'calibration_step{step + 1}'
            with paddle.no_grad():
                for inp, target in self.calib_loader:
                    inp = inp.cuda(blocking=True)
                    self.net(inp)
        for name, module in self.wrapped_modules.items():
            module.mode = 'quant_forward'
        paddle.device.cuda.empty_cache()
        print('sequential calibration finished')

    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        print(f'Start calibration step=1')
        for name, module in self.wrapped_modules.items():
            if hasattr(module, 'calibrated'):
                module.mode = 'raw'
            else:
                module.mode = f'calibration_step1'
        with paddle.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.cuda(blocking=True)
                self.net(inp)
        for name, module in self.wrapped_modules.items():
            if hasattr(module, 'calibrated'):
                continue
            else:
                module.mode = f'calibration_step2'
                with paddle.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(blocking=
                            True), module.raw_input[1].cuda(blocking=True))
                    paddle.device.cuda.empty_cache()
        for name, module in self.wrapped_modules.items():
            module.mode = 'quant_forward'
        paddle.device.cuda.empty_cache()
        print('calibration finished')

    def quant_calib(self):
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f'prepare parallel calibration for {calib_layers}')
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f'prepare parallel calibration for {calib_layers}')
        print('start calibration')
        q = tqdm(self.wrapped_modules.items(), desc='Brecq')
        for name, module in q:
            q.set_postfix_str(name)
            """
            if name == "head":
                import pdb; pdb.set_trace()
            else:
                continue
            """
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_post_hook(hook=
                    linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_post_hook(hook=
                    conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_post_hook(hook=
                    matmul_forward_hook))
            for inp, target in self.calib_loader:
                for batch_st in range(0, self.calib_loader.batch_size, self
                    .batch_size):
                    self.net.clear_gradients(set_to_zero=False)
                    inp_ = inp[batch_st:batch_st + self.batch_size].cuda(
                        blocking=True)
                    self.net(inp_)
                del inp, target
                paddle.device.cuda.empty_cache()
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = paddle.concat(x=module.raw_input, axis=0)
                module.raw_out = paddle.concat(x=module.raw_out, axis=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = paddle.concat(x=module.raw_input, axis=0)
                module.raw_out = paddle.concat(x=module.raw_out, axis=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [paddle.concat(x=_, axis=0) for _ in
                    module.raw_input]
                module.raw_out = paddle.concat(x=module.raw_out, axis=0)
            for hook in hooks:
                hook.remove()
            with paddle.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                paddle.device.cuda.empty_cache()
            if self.sequential:
                module.mode = 'quant_forward'
            else:
                module.mode = 'raw'
        for name, module in self.wrapped_modules.items():
            module.mode = 'quant_forward'
        print('calibration finished')


def grad_hook(module, grad_input, grad_output):
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output.cpu().detach())

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[], []]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())


""" def model_open_last_calibrate(net):
    for m in net.sublayers():
        if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
            m.last_calibrate = True """



class HessianQuantCalibrator(QuantCalibrator):
    """
    Modularization of hessian_quant_calib in PaddlePaddle.

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.batch_size = batch_size

    def quant_calib(self):
        """
        An implementation of original hessian calibration in PaddlePaddle.
        """

        calib_layers=[]
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with paddle.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp)
                raw_pred_softmax = F.softmax(raw_pred, axis=-1).detach()
            paddle.device.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        
        for name, module in q:
            print(f"calibrating {name} ...")
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric") and module.metric == "hessian":
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0, self.calib_loader.batch_size, self.batch_size):
                    self.net.clear_gradients()
                    inp_ = inp[batch_st:batch_st + self.batch_size].cuda()
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, axis=-1), raw_pred_softmax[batch_st:batch_st + self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                paddle.device.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = paddle.concat(module.raw_input, axis=0)
                module.raw_out = paddle.concat(module.raw_out, axis=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = paddle.concat(module.raw_input, axis=0)
                module.raw_out = paddle.concat(module.raw_out, axis=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [paddle.concat(_, axis=0) for _ in module.raw_input]
                module.raw_out = paddle.concat(module.raw_out, axis=0)
            if hasattr(module, "metric") and module.metric == "hessian":
                module.raw_grad = paddle.concat(module.raw_grad, axis=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with paddle.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                paddle.device.cuda.empty_cache()
            
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")


    def batching_quant_calib(self):
        
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        # get raw_pred as target distribution 
        with paddle.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp)
                raw_pred_softmax = F.softmax(raw_pred, axis=-1).detach()
            paddle.device.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Iter")
        
        """ import re
        for name, module in q:
            print(name)
            try:
                block_num = re.findall(r"\d+", name)[0]
                try:
                    self.blockwise_modules[block_num].append(name)
                except:
                    self.blockwise_modules[block_num] = [name]
            except:
                if name == "patch_embedding.patch_embedding":
                    self.blockwise_modules['conv'] = [name]
                elif name == "classifier":
                    self.blockwise_modules['head'] = [name]
        """

        print(f"wrapped_modules={self.wrapped_modules}")
        print(f"blockwise_modules={self.blockwise_modules}")

        q = tqdm(self.wrapped_modules.items(), desc="hessian")
        for name, module in q:
            q.set_postfix_str(name)

            # for key in self.blockwise_modules:
            #     for name in reversed(self.blockwise_modules[key]):
                    
            #         print("name=" + str(name))
            #         module = self.wrapped_modules[name]
                
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_post_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_post_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_post_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_forward_post_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for i, (inp, target) in enumerate(self.calib_loader):
                for batch_st in range(0, self.calib_loader.batch_size, self.batch_size):
                    self.net.clear_gradients()
                    inp_ = inp[batch_st:batch_st + self.batch_size].cuda()
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, axis=-1), raw_pred_softmax[batch_st:batch_st + self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                paddle.device.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = paddle.concat(module.raw_input, axis=0)
                module.raw_out = paddle.concat(module.raw_out, axis=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = paddle.concat(module.raw_input, axis=0)
                module.raw_out = paddle.concat(module.raw_out, axis=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [paddle.concat(_, axis=0) for _ in module.raw_input]
                module.raw_out = paddle.concat(module.raw_out, axis=0)
            if hasattr(module, "metric"):
                module.raw_grad = paddle.concat(module.raw_grad, axis=0)
                # print("raw_grad.shape=" + str(module.raw_grad.shape))
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with paddle.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                paddle.device.cuda.empty_cache()
            
            try:
                print(f"scaling factor: A_interval: {module.A_interval}, B_interval: {module.B_interval} for matmul")
            except:
                print(f"scaling factor: a_interval: {module.a_interval}, w_interval: {module.w_interval} for linear")

            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
            
        print("calibration finished")


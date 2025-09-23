import paddle
import paddle.nn.functional as F
from .quant_layer import QuantModule, Union
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock


def save_inp_oup_data(
    model: QuantModel,
    layer: Union[QuantModule, BaseQuantBlock],
    cali_data: paddle.Tensor,
    wq: bool = False,
    aq: bool = False,
    batch_size: int = 32,
    keep_gpu: bool = True,
    input_prob: bool = False,
):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    """
    device = next(model.parameters()).place
    get_inp_out = GetLayerInpOut(
        model, layer, device=device, wq=wq, aq=aq, input_prob=input_prob
    )
    cached_batches = []

    for i in range(int(cali_data.shape[0] / batch_size)):
        if input_prob:
            cur_inp, cur_out, cur_sym = get_inp_out(
                [_[i * batch_size : (i + 1) * batch_size] for _ in cali_data]
            )
            cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_sym.cpu()))
        else:
            cur_inp, cur_out = get_inp_out(
                [_[i * batch_size : (i + 1) * batch_size] for _ in cali_data]
            )
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
    
    cached_inps = paddle.concat([x[0] for x in cached_batches])
    cached_outs = paddle.concat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = paddle.concat([x[2] for x in cached_batches])
    
    paddle.cuda.empty_cache()

    if keep_gpu:
        cached_inps = cached_inps.cuda()
        cached_outs = cached_outs.cuda()
        if input_prob:
            cached_sym = cached_sym.cuda()

    if input_prob:
        return (cached_inps, cached_sym), cached_outs
    return (cached_inps,), cached_outs


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(
        self,
        model: QuantModel,
        layer: Union[QuantModule, BaseQuantBlock],
        device: paddle.device,
        wq: bool = False,
        aq: bool = False,
        input_prob: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.wq = wq
        self.aq = aq
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True
        )
        self.input_prob = input_prob

    def __call__(self, model_input):
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with paddle.no_grad():
            try:
                _ = self.model(*[_.to(self.device) for _ in model_input])
            except StopForwardException:
                pass
            if self.input_prob:
                input_sym = self.data_saver.input_store[0].detach()
            if self.wq or self.aq:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=self.wq, act_quant=self.aq)
                try:
                    _ = self.model(*[_.to(self.device) for _ in model_input])
                except StopForwardException:
                    pass

            self.data_saver.store_output = True
        handle.remove()

        if self.input_prob:
            return (
                self.data_saver.input_store[0].detach(),
                self.data_saver.output_store.detach(),
                input_sym,
            )
        return (
            self.data_saver.input_store[0].detach(),
            self.data_saver.output_store.detach(),
        )


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(
        self,
        model: QuantModel,
        layer: Union[QuantModule, BaseQuantBlock],
        device: paddle.device,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with paddle.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(
                    F.log_softmax(out_q, axis=1),
                    F.softmax(out_fp, axis=1),
                    reduction="batchmean",
                )
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(
    model: QuantModule,
    layer: Union[QuantModule, BaseQuantBlock],
    act_quant: bool = False,
):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break

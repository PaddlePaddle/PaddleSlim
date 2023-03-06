import paddle
from paddle.nn import ParameterList

class PropagateQuantTensor(object):
    QUANTIZED_THROUGH_ESTIMATION = 0
    STRAIGHT_THROUGH_ESTIMATION = 1
    NO_QUANTIZATION_ESTIMATION = 2

    def __init__(self, func, quant_estimation_type, copy_always=True):
        self.func = func
        self.quant_estimation_type = quant_estimation_type
        self.copy_always = copy_always

    def __call__(self, x, *args, **kwargs):
        if self.func is not None:
            y = self.func(x, *args, **kwargs)
        else:
            y = args[0]

        if self.quant_estimation_type == self.QUANTIZED_THROUGH_ESTIMATION:
            # backprop will flow through the backward function
            # as the quantized tensor is forwarded as it is.
            return y
        elif self.quant_estimation_type == self.STRAIGHT_THROUGH_ESTIMATION:
            # forward the quantized data, but in the container x_copy
            # here a copy of x is made so that the original is not altered (x is a reference in Python).
            # the backward will directly flow though x_copy and to x instead of going through y
            # copy_always can be set to False avoid this copy if possible.
            x_copy = x.clone() if self.copy_always or isinstance(x, ParameterList) else x
            x_copy.set_value(y)
            return x_copy
        elif self.quant_estimation_type == self.NO_QUANTIZATION_ESTIMATION:
            # beware! no quantization performed in this case.
            return x
        else:
            assert False, f'unknown quant_estimation_type {self.quant_estimation_type}'


class PropagateQuantTensorQTE(PropagateQuantTensor):
    def __init__(self, func):
        # QUANTIZED_THROUGH_ESTIMATION: backprop will flow through
        # the backward functions wrapped in this class
        super().__init__(func, quant_estimation_type=PropagateQuantTensor.QUANTIZED_THROUGH_ESTIMATION)


class PropagateQuantTensorSTE(PropagateQuantTensor):
    def __init__(self, func):
        # STRAIGHT_THROUGH_ESTIMATION: backprop will NOT flow through
        # the backward functions wrapped in this class
        super().__init__(func, quant_estimation_type=PropagateQuantTensor.STRAIGHT_THROUGH_ESTIMATION)

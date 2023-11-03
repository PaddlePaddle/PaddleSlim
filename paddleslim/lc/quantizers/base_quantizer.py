import paddle


class BaseQuantizer():
    def quantize(self, x: paddle.Tensor):
        raise NotImplementedError()

    def dequantize(self, x: paddle.Tensor):
        raise NotImplementedError()

    def matmul(self, x: paddle.Tensor, y: paddle.Tensor, bias: paddle.Tensor):
        raise NotImplementedError()

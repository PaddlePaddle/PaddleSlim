import copy

import paddle
from smoothquant.fake_quant import W8A8Linear
from smoothquant.utils import clip_matrix

############################## 相关utils函数，如下 ##############################

def _Tensor_reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])

setattr(paddle.Tensor, "reshape", _Tensor_reshape)
############################## 相关utils函数，如上 ##############################



class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.place
        self.rows = tuple(layer.weight.data.shape)[1]
        self.columns = tuple(layer.weight.data.shape)[0]
        self.inp_sum = None
        self.inp_num = 0
        self.scaler_row = paddle.zeros(shape=self.columns)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(tuple(inp.shape)) == 2:
            inp = inp.unsqueeze(axis=0)
        tmp = tuple(inp.shape)[0]
        if isinstance(self.layer, paddle.nn.Linear) or isinstance(
            self.layer, W8A8Linear
        ):
            if len(tuple(inp.shape)) == 3:
                inp = inp.reshape((-1, tuple(inp.shape)[-1]))
            inp = inp.t()
        if self.inp_sum is None:
            self.inp_sum = copy.deepcopy(inp.t())
        else:
            self.inp_sum += copy.deepcopy(inp.t())
        self.inp_num += 1
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.astype("float32")
        # self.scaler_row += paddle.linalg.norm(x=inp, p=2, axis=1) ** 2 / self.nsamples
        # self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        # 用 elementwise add（避免 linalg.norm 的潜在兼容问题）
        # import pdb
        # import pdb; pdb.set_trace()
        self.scaler_row = paddle.add(self.scaler_row, paddle.sum(inp * inp, axis=1) / float(self.nsamples))
        # self.scaler_row = self.scaler_row + paddle.sum(inp * inp, axis=1) / float(self.nsamples)

        



import sys
import paddle
import paddle.nn as nn
import math

activations = {'ReLU': paddle.nn.ReLU, 
               'Hardtanh': paddle.nn.Hardtanh}

class BinaryQuantize(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        # Save input tensor for backward pass
        ctx.save_for_backward(input)
        # Quantize using sign function
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from forward pass
        input, = ctx.saved_tensor()
        grad_input = grad_output
        # Apply gradient masking for values out of [-1, 1] range
        grad_input[paddle.greater_than(input, paddle.to_tensor(1.))] = 0
        grad_input[paddle.less_than(input, paddle.to_tensor(-1.))] = 0
        return grad_input

class BinaryQuantize_Vanilla(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, scale=None):
        # Save input tensor for backward pass
        ctx.save_for_backward(input)
        # Quantize using sign function
        out = paddle.sign(input)
        if scale is not None:
            out = out * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from forward pass
        input, = ctx.saved_tensor()
        grad_input = grad_output
        # Apply gradient masking for values out of [-1, 1] range
        grad_input[paddle.greater_than(input, paddle.to_tensor(1.))] = 0
        grad_input[paddle.less_than(input, paddle.to_tensor(-1.))] = 0
        return grad_input, None
    

class BiLinearVanilla(paddle.nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(BiLinearVanilla, self).__init__(in_features, out_features,
            bias_attr=bias)
        self.output_ = None

    def forward(self, input):
        bw = self.weight
        ba = input
        sw = bw.abs().mean(axis=-1).reshape([-1, 1]).detach()
        bw = BinaryQuantize_Vanilla().apply(bw, sw)
        ba = BinaryQuantize().apply(ba)
        output = paddle.nn.functional.linear(x=ba, weight=bw, bias=self.bias)
        self.output_ = output
        return output


biLinears = {(False): paddle.nn.Linear, 'Vanilla': BiLinearVanilla}


class BiConv1dVanilla(paddle.nn.Conv1D):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(BiConv1dVanilla, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, padding_mode=padding_mode, bias_attr=bias)

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        sw = bw.abs().reshape([bw.shape[0], bw.shape[1], -1]).mean(axis=-1).reshape([bw.shape[0], bw.shape[1], 1]).detach()
        bw = BinaryQuantize_Vanilla().apply(bw, sw)
        ba = BinaryQuantize().apply(ba)
        if self._padding_mode == 'circular':
            expanded_padding = (self._padding[0] + 1) // 2, self._padding[0] // 2
            return paddle.nn.functional.conv1d(x=paddle.nn.functional.pad(x
                =ba, pad=expanded_padding, mode='circular',
                pad_from_left_axis=False), weight=bw, bias=self.bias,
                stride=self._stride, padding=_single(0), dilation=self.
                _dilation, groups=self._groups)
        return paddle.nn.functional.conv1d(x=ba, weight=bw, bias=self.bias,
            stride=self._stride, padding=self._padding, dilation=self._dilation, groups=self._groups)


biConv1ds = {(False): paddle.nn.Conv1D, 'Vanilla': BiConv1dVanilla}


class BiConv2dVanilla(paddle.nn.Conv2D):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(BiConv2dVanilla, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, padding_mode=padding_mode, bias_attr=bias)

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        sw = bw.abs().reshape([bw.shape[0], bw.shape[1], -1]).mean(axis=-1).reshape([bw.shape[0], bw.shape[1], 1, -1]).detach()
        bw = BinaryQuantize_Vanilla().apply(bw, sw)
        ba = BinaryQuantize().apply(ba)
        if self._padding_mode == 'circular':
            expanded_padding = (self._padding[0] + 1) // 2, self._padding[0] // 2
            return paddle.nn.functional.conv2d(x=paddle.nn.functional.pad(x
                =ba, pad=expanded_padding, mode='circular',
                pad_from_left_axis=False), weight=bw, bias=self.bias,
                stride=self._stride, padding=_pair(0), dilation=self.
                _dilation, groups=self._groups)
        return paddle.nn.functional.conv2d(x=ba, weight=bw, bias=self.bias,
            stride=self._stride, padding=self._padding, dilation=self.
            _dilation, groups=self._groups)


biConv2ds = {(False): paddle.nn.Conv2D, 'Vanilla': BiConv2dVanilla}



def Modify(module: nn.Layer, method='Sign', id=-1, first=-1, last=-1):
    id = 0 if id == -1 else id
    temp = {}
    if method != False:
        for name, child_module in module.named_sublayers():
            if isinstance(child_module, nn.LayerList):
                for child_child_module in child_module:
                    _, id = Modify(child_child_module, method=method, id=id, first=first, last=last)
            else:
                _, id = Modify(child_module, method=method, id=id, first=first, last=last)
                
                if isinstance(child_module, nn.Linear):
                    id += 1
                    if id <= first or id >= last:
                        continue
                    in_feature = child_module.weight.shape[0]
                    out_feature = child_module.weight.shape[1]
                    new_layer = biLinears[method](in_feature, out_feature)
                    new_layer.weight = child_module.weight
                    new_layer.bias = child_module.bias

                    temp[name] = new_layer
                    # module._sub_layers[name] = new_layer
                
                elif isinstance(child_module, nn.Conv1D):
                    id += 1
                    if id <= first or id >= last:
                        continue
                    new_layer = biConv1ds[method](in_channels=child_module._in_channels,
                                                  out_channels=child_module._out_channels,
                                                  kernel_size=child_module._kernel_size,
                                                  stride=child_module._stride,
                                                  padding=child_module._padding,
                                                  dilation=child_module._dilation,
                                                  groups=child_module._groups,
                                                  bias=child_module._bias_attr)
                    new_layer.weight = child_module.weight
                    new_layer.bias = child_module.bias

                    temp[name] = new_layer
                    # module._sub_layers[name] = new_layer

                elif isinstance(child_module, nn.Conv2D):
                    id += 1
                    if id <= first or id >= last:
                        continue
                    new_layer = biConv2ds[method](in_channels=child_module._in_channels,
                                                  out_channels=child_module._out_channels,
                                                  kernel_size=child_module._kernel_size,
                                                  stride=child_module._stride,
                                                  padding=child_module._padding,
                                                  dilation=child_module._dilation,
                                                  groups=child_module._groups,
                                                  bias=child_module._bias_attr)
                    new_layer.weight = child_module.weight
                    new_layer.bias = child_module.bias
                    
                    temp[name] = new_layer
                    # module._sub_layers[name] = new_layer、

        for name, child_module in temp.items():
            module._sub_layers[name] = child_module

    return module, id
    
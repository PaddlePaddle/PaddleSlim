import sys
import paddle


class StraightThrough(paddle.nn.Layer):

    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = paddle.sqrt(x=y_var + bn_module.eps)
    w_view = conv_module.out_channels, 1, 1, 1
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def reset_bn(module: paddle.nn.BatchNorm2D):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1 - module.eps)
    if module.affine:
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(module.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(module.bias)


def is_bn(m):
    return isinstance(m, paddle.nn.BatchNorm2D) or isinstance(m, paddle.nn.
        BatchNorm1D)


def is_absorbing(m):
    return isinstance(m, paddle.nn.Conv2D) or isinstance(m, paddle.nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def search_fold_and_reset_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
        else:
            search_fold_and_reset_bn(m)
        prev = m

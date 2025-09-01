import paddle
from paddle import nn


def round_ste(x: paddle.Tensor):
    return (x.round() - x).detach() + x


def floor_ste(x: paddle.Tensor):
    return (x.floor() - x).detach() + x


def ceil_ste(x: paddle.Tensor):
    return (x.ceil() - x).detach() + x

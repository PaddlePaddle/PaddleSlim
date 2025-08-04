import paddle
import paddle.nn as nn

def weight_init(m, **kwargs):
    # weight initialization
    if isinstance(m, nn.Conv2D):
        # Using kaiming_normal for Conv2D
        nn.initializer.KaimingNormal()(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.initializer.Zeros()(m.bias)
    elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
        # BatchNorm2D or GroupNorm
        nn.initializer.Ones()(m.weight)
        nn.initializer.Zeros()(m.bias)
    elif isinstance(m, nn.Linear):
        # Linear layer initialization
        nn.initializer.Normal(mean=0.0, std=0.01)(m.weight)
        nn.initializer.Zeros()(m.bias)


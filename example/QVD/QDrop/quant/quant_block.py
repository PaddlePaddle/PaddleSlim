import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .quant_layer import QuantModule, UniformAffineQuantizer
from QDrop.models.resnet import BasicBlock, Bottleneck
from QDrop.models.regnet import ResBottleneckBlock
from QDrop.models.mobilenetv2 import InvertedResidual
from QDrop.models.mnasnet import _InvertedResidual
from magicanimate.models.resnet import InflatedConv3d
from einops import rearrange
from ppdiffusers.models.lora import LoRACompatibleLinear
import time


class BaseQuantBlock(nn.Layer):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self):
        super(BaseQuantBlock, self).__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.sublayers():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        self.activation_function = basic_block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantBottleneck, self).__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantResBottleneckBlock, self).__init__()
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        self.proj_block = bottleneck.proj_block
        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantInvertedResidual, self).__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class _QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, _inv_res: _InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(_QuantInvertedResidual, self).__init__()

        self.apply_residual = _inv_res.apply_residual
        self.conv = nn.Sequential(
            QuantModule(_inv_res.layers[0], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[3], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[6], weight_quant_params, act_quant_params, disable_act_quant=True),
        )
        self.conv[0].activation_function = nn.ReLU()
        self.conv[1].activation_function = nn.ReLU()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInflatedConv3d(BaseQuantBlock):
    def __init__(self, i3d: InflatedConv3d, weight_quant_params, act_quant_params):
        super(QuantInflatedConv3d, self).__init__()

        self.conv1 = QuantModule(i3d, weight_quant_params, act_quant_params)

    def forward(self, x):
        video_length = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv1(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        return x


class QuantLoRACompatibleLinear(BaseQuantBlock):
    def __init__(self, llinear: LoRACompatibleLinear, weight_quant_params, act_quant_params):
        super(QuantLoRACompatibleLinear, self).__init__()
        self.ori_linear = llinear
        self.linear = QuantModule(llinear, weight_quant_params, act_quant_params)

    def forward(self, hidden_states, scale: float = 1.0):
        out = self.linear(hidden_states)
        return out


specials = {
    LoRACompatibleLinear: QuantLoRACompatibleLinear,
    InflatedConv3d: QuantInflatedConv3d,
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,
}

from .base_models import MyNetwork
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class LambdaLayer(nn.Layer):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.conv1 = nn.Conv2D(in_planes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential()
        self.stride = stride

        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.planes != self.in_planes:
            if self.stride != 1:
                self.downsample = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, (self.planes-self.in_planes)//2, self.planes-self.in_planes-(self.planes-self.in_planes)//2,
                                    0, 0, 0, 0), "constant", 0))
            else:
                self.downsample = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, (self.planes-self.in_planes) // 2,
                                     self.planes-self.in_planes-(self.planes-self.in_planes)//2, 0, 0, 0, 0), "constant", 0))


        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar(MyNetwork):
    def __init__(self, depth=20, num_classes=10, cfg=None, cutout=False):
        super(ResNetCifar, self).__init__()
        cfg_base = []
        n = (depth-2) // 6

        for i in [16, 32, 64]:
            for j in range(n):
                cfg_base.append(i)

        if cfg is None:
            cfg = cfg_base

        num_blocks = []
        if depth == 20:
            num_blocks = [3, 3, 3]

        block = BasicBlock

        self.cfg_base = cfg_base
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_planes = 16

        conv1 = nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, bias_attr=False)
        bn1 = nn.BatchNorm2D(16)
        self.conv_bn = nn.Sequential(conv1, bn1)

        self.layer1 = self._make_layer(block, cfg[0:n], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[n:2*n], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2*n:], num_blocks[2], stride=2)

        self.pool = nn.AdaptiveAvgPool2D(1)
        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(len(strides)):
            layers.append(('block_%d' % i, block(self.in_planes, planes[i], strides[i])))
            self.in_planes = planes[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).flatten(1)
        out = self.linear(out)

        return out

    def feature_extract(self, x):
        tensor = []
        out = F.relu(self.conv_bn(x))
        for i in [self.layer1, self.layer2, self.layer3]:
            for _layer in i:
                out = _layer(out)
                if type(_layer) is BasicBlock:
                    tensor.append(out)

        return tensor

    def cfg2params(self, cfg):
        params = 0
        params += (3 * 3 * 3 * 16 + 16 * 2) # conv1+bn1
        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                params += (in_c * 3 * 3 * c + 2 * c + c * 3 * 3 * c + 2 * c) # per block params
                if in_c != c:
                    params += in_c * c # shortcut
                in_c = c
                cfg_idx += 1
        params += (self.cfg[-1] + 1) * self.num_classes # fc layer
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        size = 32
        flops = 0
        flops += (3 * 3 * 3 * 16 * 32 * 32 + 16 * 32 * 32 * 4) # conv1+bn1
        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                flops += (in_c * 3 * 3 * c * size * size + c * size * size * 4 + c * 3 * 3 * c * size * size + c * size * size * 4) # per block flops
                if in_c != c:
                    flops += in_c * c * size * size # shortcut
                in_c = c
                cfg_idx += 1
        flops += (2 * self.cfg[-1] + 1) * self.num_classes # fc layer
        return flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        size = 32
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]

        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                if i==0 and j==0:
                    flops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4 + in_c * 3 * 3 * c * size * size)
                    flops_squarecfg[cfg_idx] += c * 3 * 3 * c * size * size
                else:
                    flops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4)
                    flops_doublecfg[cfg_idx-1][cfg_idx] += in_c * 3 * 3 * c * size * size
                    flops_doublecfg[cfg_idx][cfg_idx-1] += in_c * 3 * 3 * c * size * size
                    flops_squarecfg[cfg_idx] += (c * 3 * 3 * c * size * size )
                if in_c != c:
                    flops_doublecfg[cfg_idx][cfg_idx-1] += in_c * c * size * size # shortcut
                    flops_doublecfg[cfg_idx-1][cfg_idx] += in_c * c * size * size
                in_c = c
                cfg_idx += 1

        flops_singlecfg[-1] += 2 * self.cfg[-1] * self.num_classes # fc layer
        return flops_singlecfg, flops_doublecfg, flops_squarecfg

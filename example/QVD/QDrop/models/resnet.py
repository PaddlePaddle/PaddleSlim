import paddle
import paddle.nn as nn
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'wide_resnet101_2']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=stride, padding=dilation, groups=groups,
        bias_attr=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(paddle.nn.Layer):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.relu2 = paddle.nn.ReLU()
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


class Bottleneck(paddle.nn.Layer):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = paddle.nn.ReLU()
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class ResNet(nn.Layer):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 deep_stem=False,
                 avg_down=False):

        super(ResNet, self).__init__()

        global BN

        BN = nn.BatchNorm2D
        norm_layer = BN

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2D(3, 32, kernel_size=3, stride=2,
                          padding=1, bias_attr=False),
                norm_layer(32),
                nn.ReLU(),
                nn.Conv2D(32, 32, kernel_size=3, stride=1,
                          padding=1, bias_attr=False),
                norm_layer(32),
                nn.ReLU(),
                nn.Conv2D(32, 64, kernel_size=3, stride=1,
                          padding=1, bias_attr=False),
            )
        else:
            self.conv1 = nn.Conv2D(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias_attr=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                nn.initializer.Constant(value=1.0)(m.weight)
                nn.initializer.Constant(value=0.0)(m.bias)
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, Bottleneck):
                    nn.initializer.Constant(value=0.0)(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.initializer.Constant(value=0.0)(m.bn2.weight)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2D(kernel_size=stride, stride=stride, ceil_mode=True, exclusive=False),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def wide_resnet50_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def wide_resnet101_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

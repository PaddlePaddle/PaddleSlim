# !/usr/bin/env python

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['mnasnet']

class _InvertedResidual(nn.Layer):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2D(in_ch, mid_ch, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(mid_ch),
            nn.ReLU(),
            # Depthwise
            nn.Conv2D(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, 
                      stride=stride, groups=mid_ch, bias_attr=False),
            nn.BatchNorm2D(mid_ch),
            nn.ReLU(),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2D(mid_ch, out_ch, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(out_ch))

    def forward(self, x):
        if self.apply_residual:
            return self.layers(x) + x
        else:
            return self.layers(x)

def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor))
    return nn.Sequential(first, *remaining)

def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor

def _get_depths(scale):
    """ Scales tensor depths as in reference MobileNet code, prefers rounding up """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * scale, 8) for depth in depths]

class MNASNet(nn.Layer):
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, scale=2.0, num_classes=1000, dropout=0.0):
        super(MNASNet, self).__init__()

        assert scale > 0.0
        self.scale = scale
        self.num_classes = num_classes
        depths = _get_depths(scale)

        layers = [
            # First layer: regular conv.
            nn.Conv2D(3, depths[0], kernel_size=3, padding=1, stride=2, bias_attr=False),
            nn.BatchNorm2D(depths[0]),
            nn.ReLU(),
            # Depthwise separable, no skip.
            nn.Conv2D(depths[0], depths[0], kernel_size=3, padding=1, stride=1, 
                      groups=depths[0], bias_attr=False),
            nn.BatchNorm2D(depths[0]),
            nn.ReLU(),
            nn.Conv2D(depths[0], depths[1], kernel_size=1, padding=0, stride=1, bias_attr=False),
            nn.BatchNorm2D(depths[1]),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], kernel_size=3, stride=2, exp_factor=3, repeats=3),
            _stack(depths[2], depths[3], kernel_size=5, stride=2, exp_factor=3, repeats=3),
            _stack(depths[3], depths[4], kernel_size=5, stride=2, exp_factor=6, repeats=3),
            _stack(depths[4], depths[5], kernel_size=3, stride=1, exp_factor=6, repeats=2),
            _stack(depths[5], depths[6], kernel_size=5, stride=2, exp_factor=6, repeats=4),
            _stack(depths[6], depths[7], kernel_size=3, stride=1, exp_factor=6, repeats=1),
            # Final mapping to classifier input.
            nn.Conv2D(depths[7], 1280, kernel_size=1, padding=0, stride=1, bias_attr=False),
            nn.BatchNorm2D(1280),
            nn.ReLU(),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.mean(axis=[2, 3])  # Global average pooling
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
                if m.bias is not None:
                    nn.initializer.Constant(0.0)(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(1.0)(m.weight)
                nn.initializer.Constant(0.0)(m.bias)
            elif isinstance(m, nn.Linear):
                nn.initializer.KaimingUniform()(m.weight)
                nn.initializer.Constant(0.0)(m.bias)

def mnasnet(**kwargs):
    model = MNASNet(**kwargs)
    return model

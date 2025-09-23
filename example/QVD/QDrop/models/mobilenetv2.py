import paddle
import math


def conv_bn(inp, oup, stride):
    return paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inp,
        out_channels=oup, kernel_size=3, stride=stride, padding=1,
        bias_attr=False), paddle.nn.BatchNorm2D(num_features=oup), paddle.
        nn.ReLU6())


def conv_1x1_bn(inp, oup):
    return paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inp,
        out_channels=oup, kernel_size=1, stride=1, padding=0, bias_attr=
        False), paddle.nn.BatchNorm2D(num_features=oup), paddle.nn.ReLU6())


class InvertedResidual(paddle.nn.Layer):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        if expand_ratio == 1:
            self.conv = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
                hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=
                stride, padding=1, groups=hidden_dim, bias_attr=False),
                paddle.nn.BatchNorm2D(num_features=hidden_dim), paddle.nn.
                ReLU6(), paddle.nn.Conv2D(in_channels=hidden_dim,
                out_channels=oup, kernel_size=1, stride=1, padding=0,
                bias_attr=False), paddle.nn.BatchNorm2D(num_features=oup))
        else:
            self.conv = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
                inp, out_channels=hidden_dim, kernel_size=1, stride=1,
                padding=0, bias_attr=False), paddle.nn.BatchNorm2D(
                num_features=hidden_dim), paddle.nn.ReLU6(), paddle.nn.
                Conv2D(in_channels=hidden_dim, out_channels=hidden_dim,
                kernel_size=3, stride=stride, padding=1, groups=hidden_dim,
                bias_attr=False), paddle.nn.BatchNorm2D(num_features=
                hidden_dim), paddle.nn.ReLU6(), paddle.nn.Conv2D(
                in_channels=hidden_dim, out_channels=oup, kernel_size=1,
                stride=1, padding=0, bias_attr=False), paddle.nn.
                BatchNorm2D(num_features=oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(paddle.nn.Layer):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0,
        dropout=0.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = paddle.nn.Sequential(*self.features)
        self.classifier = paddle.nn.Sequential(paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=self.last_channel, out_features=
            n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(axis=[2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, paddle.nn.BatchNorm2D):
                m.weight.data.fill_(value=1)
                m.bias.data.zero_()
            elif isinstance(m, paddle.nn.Linear):
                n = m.weight.shape[1]
                m.weight.data.normal_(mean=0, std=0.01)
                m.bias.data.zero_()


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNetV2 model.
    """
    model = MobileNetV2(**kwargs)
    return model

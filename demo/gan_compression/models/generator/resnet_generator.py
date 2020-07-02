import functools
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose
from paddle.nn.layer import ReLU, Pad2D
from paddleslim.models.dygraph.modules import ResnetBlock


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self,
                 input_channel,
                 output_nc,
                 ngf,
                 norm_layer=InstanceNorm,
                 dropout_rate=0,
                 n_blocks=6,
                 padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == InstanceNorm
        else:
            use_bias = norm_layer == InstanceNorm

        self.model = fluid.dygraph.LayerList([
            Pad2D(
                paddings=[3, 3, 3, 3], mode="reflect"), Conv2D(
                    input_channel,
                    int(ngf),
                    filter_size=7,
                    padding=0,
                    bias_attr=use_bias), norm_layer(ngf), ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.model.extend([
                Conv2D(
                    ngf * mult,
                    ngf * mult * 2,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(ngf * mult * 2), ReLU()
            ])

        mult = 2**n_downsampling
        for i in range(n_blocks):
            self.model.extend([
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.model.extend([
                Conv2DTranspose(
                    ngf * mult,
                    int(ngf * mult / 2),
                    filter_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), Pad2D(
                        paddings=[0, 1, 0, 1], mode='constant', pad_value=0.0),
                norm_layer(int(ngf * mult / 2)), ReLU()
            ])

        self.model.extend([Pad2D(paddings=[3, 3, 3, 3], mode="reflect")])
        self.model.extend([Conv2D(ngf, output_nc, filter_size=7, padding=0)])

    def forward(self, inputs):
        y = fluid.layers.clamp(inputs, min=-1.0, max=1.0)
        for sublayer in self.model:
            y = sublayer(y)
        y = fluid.layers.tanh(y)
        return y

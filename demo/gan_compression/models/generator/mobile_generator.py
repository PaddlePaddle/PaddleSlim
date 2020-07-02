import functools
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose
from paddle.nn.layer import ReLU, Pad2D
from paddleslim.models.dygraph.modules import MobileResnetBlock

use_cudnn = False


class MobileResnetGenerator(fluid.dygraph.Layer):
    def __init__(self,
                 input_channel,
                 output_nc,
                 ngf,
                 norm_layer=InstanceNorm,
                 dropout_rate=0,
                 n_blocks=9,
                 padding_type='reflect'):
        super(MobileResnetGenerator, self).__init__()
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
                    use_cudnn=use_cudnn,
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
                    use_cudnn=use_cudnn,
                    bias_attr=use_bias), norm_layer(ngf * mult * 2), ReLU()
            ])

        mult = 2**n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            self.model.extend([
                MobileResnetBlock(
                    ngf * mult,
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_blocks2):
            self.model.extend([
                MobileResnetBlock(
                    ngf * mult,
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_blocks3):
            self.model.extend([
                MobileResnetBlock(
                    ngf * mult,
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            output_size = (i + 1) * 128
            self.model.extend([
                Conv2DTranspose(
                    ngf * mult,
                    int(ngf * mult / 2),
                    filter_size=3,
                    output_size=output_size,
                    stride=2,
                    padding=1,
                    use_cudnn=use_cudnn,
                    bias_attr=use_bias), norm_layer(int(ngf * mult / 2)),
                ReLU()
            ])

        self.model.extend([Pad2D(paddings=[3, 3, 3, 3], mode="reflect")])
        self.model.extend([Conv2D(ngf, output_nc, filter_size=7, padding=0)])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.model:
            y = sublayer(y)
        y = fluid.layers.tanh(y)
        return y

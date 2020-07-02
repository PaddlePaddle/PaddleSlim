import functools
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose
from paddle.nn.layer import ReLU, Pad2D
from paddleslim.models.dygraph.modules import SeparableConv2D, MobileResnetBlock


class SubMobileResnetGenerator(fluid.dygraph.Layer):
    def __init__(self,
                 input_channel,
                 output_nc,
                 config,
                 norm_layer=InstanceNorm,
                 dropout_rate=0,
                 n_blocks=9,
                 padding_type='reflect'):
        super(SubMobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == InstanceNorm
        else:
            use_bias = norm_layer == InstanceNorm

        self.model = fluid.dygraph.LayerList([
            Pad2D(
                paddings=[3, 3, 3, 3], mode="reflect"), Conv2D(
                    input_channel,
                    config['channels'][0],
                    filter_size=7,
                    padding=0,
                    use_cudnn=use_cudnn,
                    bias_attr=use_bias), norm_layer(config['channels'][0]),
            ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            in_c = config['channels'][i]
            out_c = config['channels'][i + 1]
            self.model.extend([
                Conv2D(
                    in_c * mult,
                    out_c * mult * 2,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    use_cudnn=use_cudnn,
                    bias_attr=use_bias), norm_layer(out_c * mult * 2), ReLU()
            ])

        mult = 2**n_downsampling

        in_c = config['channels'][2]
        for i in range(n_blocks):
            if len(config['channels']) == 6:
                offset = 0
            else:
                offset = i // 3
            out_c = config['channels'][offset + 3]
            self.model.extend([
                MobileResnetBlock(
                    in_c * mult,
                    out_c * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        if len(config['channels']) == 6:
            offset = 4
        else:
            offset = 6
        for i in range(n_downsampling):
            out_c = config['channels'][offset + i]
            mult = 2**(n_downsampling - i)
            output_size = (i + 1) * 128
            self.model.extend([
                Conv2DTranspose(
                    in_c * mult,
                    int(out_c * mult / 2),
                    filter_size=3,
                    output_size=output_size,
                    stride=2,
                    padding=1,
                    use_cudnn=use_cudnn,
                    bias_attr=use_bias), norm_layer(int(out_c * mult / 2)),
                ReLU()
            ])
            in_c = out_c

        self.model.extend([Pad2D(paddings=[3, 3, 3, 3], mode="reflect")])
        self.model.extend([Conv2D(in_c, output_nc, filter_size=7, padding=0)])

    def forward(self, inputs):
        y = tensor.clamp(input, min=-1, max=1)
        for sublayer in self.model:
            y = sublayer(y)
        y = fluid.layers.tanh(y)
        return y

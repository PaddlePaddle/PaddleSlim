import functools
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid.dygraph.nn import BatchNorm, InstanceNorm, Dropout
from paddle.nn.layer import ReLU, Pad2D
from paddleslim.core.layers import SuperConv2D, SuperConv2DTranspose, SuperSeparableConv2D, SuperInstanceNorm


class SuperMobileResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(SuperMobileResnetBlock, self).__init__()
        self.conv_block = fluid.dygraph.LayerList([])
        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode="reflect")])
        elif padding_type == 'replicate':
            self.conv_block.extend([Pad2D(paddings=[1, 1, 1, 1], mode="edge")])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SuperSeparableConv2D(
                num_channels=dim,
                num_filters=dim,
                filter_size=3,
                stride=1,
                padding=p), norm_layer(dim), ReLU()
        ])
        self.conv_block.extend([Dropout(dropout_rate)])

        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode="reflect")])
        elif padding_type == 'replicate':
            self.conv_block.extend([Pad2D(paddings=[1, 1, 1, 1], mode="edge")])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SuperSeparableConv2D(
                num_channels=dim,
                num_filters=dim,
                filter_size=3,
                stride=1,
                padding=p), norm_layer(dim)
        ])

    def forward(self, input, config):
        x = input
        cnt = 0
        for sublayer in self.conv_block:
            if isinstance(sublayer, SuperSeparableConv2D):
                if cnt == 1:
                    config['channel'] = input.shape[1]
                x = sublayer(x, config)
                cnt += 1
            else:
                x = sublayer(x)
        out = input + x
        return out


class SuperMobileResnetGenerator(fluid.dygraph.Layer):
    def __init__(self,
                 input_channel,
                 output_nc,
                 ngf,
                 norm_layer=InstanceNorm,
                 dropout_rate=0,
                 n_blocks=6,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super(SuperMobileResnetGenerator, self).__init__()
        use_bias = norm_layer == InstanceNorm

        if norm_layer.func == InstanceNorm or norm_layer == InstanceNorm:
            norm_layer = SuperInstanceNorm
        else:
            raise NotImplementedError

        self.model = fluid.dygraph.LayerList([])
        self.model.extend([
            Pad2D(
                paddings=[3, 3, 3, 3], mode="reflect"), SuperConv2D(
                    input_channel,
                    ngf,
                    filter_size=7,
                    padding=0,
                    bias_attr=use_bias), norm_layer(ngf), ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.model.extend([
                SuperConv2D(
                    ngf * mult,
                    ngf * mult * 2,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(int(ngf * mult * 2)),
                ReLU()
            ])

        mult = 2**n_downsampling
        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            self.model.extend([
                SuperMobileResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_blocks2):
            self.model.extend([
                SuperMobileResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_blocks3):
            self.model.extend([
                SuperMobileResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            output_size = (i + 1) * 128
            #### torch:out_padding = 1 => paddle:deconv + pad
            self.model.extend([
                SuperConv2DTranspose(
                    ngf * mult,
                    int(ngf * mult / 2),
                    filter_size=3,
                    output_size=output_size,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(int(ngf * mult / 2)),
                ReLU()
            ])

        self.model.extend([Pad2D(paddings=[3, 3, 3, 3], mode="reflect")])
        self.model.extend(
            [SuperConv2D(
                ngf, output_nc, filter_size=7, padding=0)])

    def forward(self, input):
        configs = self.configs
        x = tensor.clamp(input, min=-1, max=1)
        cnt = 0
        for i in range(0, 10):
            sublayer = self.model[i]
            if isinstance(sublayer, SuperConv2D):
                channel = configs['channels'][cnt] * (2**cnt)
                config = {'channel': channel}
                x = sublayer(x, config)
                cnt += 1
            else:
                x = sublayer(x)

        for i in range(3):
            for j in range(10 + i * 3, 13 + i * 3):
                if len(configs['channels']) == 6:
                    channel = configs['channels'][3] * 4
                else:
                    channel = configs['channels'][i + 3] * 4
                config = {'channel': channel}
                sublayer = self.model[j]
                x = sublayer(x, config)

        cnt = 2
        for i in range(19, 27):
            sublayer = self.model[i]
            if isinstance(sublayer, SuperConv2DTranspose):
                cnt -= 1
                if len(configs['channels']) == 6:
                    channel = configs['channels'][5 - cnt] * (2**cnt)
                else:
                    channel = configs['channels'][7 - cnt] * (2**cnt)
                config = {'channel': channel}
                x = sublayer(x, config)
            elif isinstance(sublayer, SuperConv2D):
                config = {'channel': sublayer._num_filters}
                x = sublayer(x, config)
            else:
                x = sublayer(x)
        x = fluid.layers.tanh(x)
        return x

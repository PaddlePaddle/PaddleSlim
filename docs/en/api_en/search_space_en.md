# search space
Search Space used in neural architecture search. Search Space is a collection of model architecture, the purpose of SANAS is to get a model which FLOPs or latency is smaller or percision is higher.

## search space which paddleslim.nas provided

#### Based on origin model architecture:
1. MobileNetV2Space<br>
&emsp; MobileNetV2's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v2.py#L29), [paper](https://arxiv.org/abs/1801.04381)

2. MobileNetV1Space<br>
&emsp; MobilNetV1's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v1.py#L29), [paper](https://arxiv.org/abs/1704.04861)

3. ResNetSpace<br>
&emsp; ResNetSpace's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py#L30), [paper](https://arxiv.org/pdf/1512.03385.pdf)


#### Based on block from different model:
1. MobileNetV1BlockSpace<br>
&emsp; MobileNetV1Block's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v1.py#L173)

2. MobileNetV2BlockSpace<br>
&emsp; MobileNetV2Block's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v2.py#L174)

3. ResNetBlockSpace<br>
&emsp; ResNetBlock's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py#L148)

4. InceptionABlockSpace<br>
&emsp; InceptionABlock's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/inception_v4.py#L140)

5. InceptionCBlockSpace<br>
&emsp; InceptionCBlock's architecture can reference: [code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/inception_v4.py#L291)


## How to use search space
1. Only need to specify the name of search space if use the space based on origin model architecture, such as configs for class SANAS is [('MobileNetV2Space')] if you want to use origin MobileNetV2 as search space.
2. Use search space paddleslim.nas provided based on block:<br>
  2.1 Use `input_size`, `output_size` and `block_num` to construct search space, such as configs for class SANAS is ('MobileNetV2BlockSpace', {'input_size': 224, 'output_size': 32, 'block_num': 10})].<br>
  2.2 Use `block_mask` to construct search space, such as configs for class SANAS is [('MobileNetV2BlockSpace', {'block_mask': [0, 1, 1, 1, 1, 0, 1, 0]})].

## How to write yourself search space
If you want to write yourself search space, you need to inherit base class named SearchSpaceBase and overwrite following functions:<br>
&emsp; 1. Function to get initial tokens(function `init_tokens`), set the initial tokens which you want, every token in tokens means index of search list, such as if tokens=[0, 3, 5], it means the list of channel of current model architecture is [8, 40, 128].
&emsp; 2. Function about the length of every token in tokens(function `range_table`), range of every token in tokens.
&emsp; 3. Function to get model architecture according to tokens(function `token2arch`), get model architecture according to tokens in the search process.

For example, how to add a search space with resnet block. New search space can NOT has the same name with existing search space.

```python
### import necessary head file
from .search_space_base import SearchSpaceBase
from .search_space_registry import SEARCHSPACE
import numpy as np

### use decorator SEARCHSPACE.register to register yourself search space to search space NameSpace
@SEARCHSPACE.register
### define a search space class inherit the base class SearchSpaceBase
class ResNetBlockSpace2(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        ### define the iterm you want to search, such as the numeber of channel, the number of convolution repeat, the size of kernel.
        ### self.filter_num represents the search list about the numeber of channel.
        self.filter_num = np.array([8, 16, 32, 40, 64, 128, 256, 512])

    ### define initial tokens, the length of initial tokens according to block_num or block_mask.
    def init_tokens(self):
        return [0] * 3 * len(self.block_mask)

    ### define the range of index in tokens.
    def range_table(self):
        return [len(self.filter_num)] * 3 * len(self.block_mask)

    ### transform tokens to model architecture.
    def token2arch(self, tokens=None):
        if tokens == None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        for i in range(len(self.block_mask)):
            self.bottleneck_params_list.append(self.filter_num[tokens[i * 3 + 0]],
                                               self.filter_num[tokens[i * 3 + 1]],
                                               self.filter_num[tokens[i * 3 + 2]],
                                               2 if self.block_mask[i] == 1 else 1)

        def net_arch(input):
            for i, layer_setting in enumerate(self.bottleneck_params_list):
                channel_num, stride = layer_setting[:-1], layer_setting[-1]
                input = self._resnet_block(input, channel_num, stride, name='resnet_layer{}'.format(i+1))

            return input

        return net_arch

    ### code to get block.
    def _resnet_block(self, input, channel_num, stride, name=None):
        shortcut_conv = self._shortcut(input, channel_num[2], stride, name=name)
        input = self._conv_bn_layer(input=input, num_filters=channel_num[0], filter_size=1, act='relu', name=name + '_conv0')
        input = self._conv_bn_layer(input=input, num_filters=channel_num[1], filter_size=3, stride=stride, act='relu', name=name + '_conv1')
        input = self._conv_bn_layer(input=input, num_filters=channel_num[2], filter_size=1, name=name + '_conv2')
        return fluid.layers.elementwise_add(x=shortcut_conv, y=input, axis=0, name=name+'_elementwise_add')

    def _shortcut(self, input, channel_num, stride, name=None):
        channel_in = input.shape[1]
        if channel_in != channel_num or stride != 1:
            return self.conv_bn_layer(input, num_filters=channel_num, filter_size=1, stride=stride, name=name+'_shortcut')
        else:
            return input

    def _conv_bn_layer(self, input, num_filters, filter_size, stride=1, padding='SAME', act=None, name=None):
        conv = fluid.layers.conv2d(input, num_filters, filter_size, stride, name=name+'_conv')
        bn = fluid.layers.batch_norm(conv, act=act, name=name+'_bn')
        return bn
```

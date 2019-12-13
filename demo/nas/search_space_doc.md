# paddleslim.nas 提供的搜索空间：

## 1. 根据原本模型结构构造搜索空间：

### 1.1 MobileNetV2Space

### 1.2 MobileNetV1Space

### 1.3 ResNetSpace


## 2. 根据相应模型的block构造搜索空间

### 2.1 MobileNetV1BlockSpace

### 2.2 MobileNetV2BlockSpace

### 2.3 ResNetBlockSpace

### 2.4 InceptionABlockSpace

### 2.5 InceptionCBlockSpace




# 自定义搜索空间(search space)

自定义搜索空间类需要继承搜索空间基类并重写以下几部分：
  1. 初始化的tokens(`init_tokens`函数)，可以设置为自己想要的tokens列表, tokens列表中的每个数字指的是当前数字在相应的搜索列表中的索引。例如本示例中若tokens=[0, 3, 5]，则代表当前模型结构搜索到的通道数为[8, 40, 128]。
  2. token中每个数字的搜索列表长度(`range_table`函数)，tokens中每个token的索引范围。
  3. 根据token产生模型结构(`token2arch`函数)，根据搜索到的tokens列表产生模型结构。

以新增reset block为例说明如何构造自己的search space。自定义的search space不能和已有的search space同名。

```python
### 引入搜索空间基类函数和search space的注册类函数
from .search_space_base import SearchSpaceBase
from .search_space_registry import SEARCHSPACE
import numpy as np

### 需要调用注册函数把自定义搜索空间注册到space space中
@SEARCHSPACE.register
### 定义一个继承SearchSpaceBase基类的搜索空间的类函数
class ResNetBlockSpace2(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        ### 定义一些实际想要搜索的内容，例如：通道数、每个卷积的重复次数、卷积核大小等等
        ### self.filter_num 代表通道数的搜索列表
        self.filter_num = np.array([8, 16, 32, 40, 64, 128, 256, 512])

    ### 定义初始化token，初始化token的长度根据传入的block_num或者block_mask的长度来得到的
    def init_tokens(self):
        return [0] * 3 * len(self.block_mask)

    ### 定义
    def range_table(self):
        return [len(self.filter_num)] * 3 * len(self.block_mask)

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

    ### 构造具体block的操作
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

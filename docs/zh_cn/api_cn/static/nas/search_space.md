# 搜索空间
搜索空间是神经网络搜索中的一个概念。搜索空间是一系列模型结构的汇集, SANAS主要是利用模拟退火的思想在搜索空间中搜索到一个比较小的模型结构或者一个精度比较高的模型结构。

## paddleslim.nas 提供的搜索空间

#### 根据初始模型结构构造搜索空间:

1. MobileNetV2Space<br>
&emsp; MobileNetV2的网络结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v2.py#L29)，[论文](https://arxiv.org/abs/1801.04381)

2. MobileNetV1Space<br>
&emsp; MobilNetV1的网络结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v1.py#L29)，[论文](https://arxiv.org/abs/1704.04861)

3. ResNetSpace<br>
&emsp; ResNetSpace的网络结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py#L30)，[论文](https://arxiv.org/pdf/1512.03385.pdf)


#### 根据相应模型的block构造搜索空间:
1. MobileNetV1BlockSpace<br>
&emsp; MobileNetV1Block的结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v1.py#L173)

2. MobileNetV2BlockSpace<br>
&emsp; MobileNetV2Block的结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v2.py#L174)

3. ResNetBlockSpace<br>
&emsp; ResNetBlock的结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py#L148)

4. InceptionABlockSpace<br>
&emsp; InceptionABlock的结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/inception_v4.py#L140)

5. InceptionCBlockSpace<br>
&emsp; InceptionCBlock结构可以参考：[代码](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/inception_v4.py#L291)


## 搜索空间使用示例

1. 使用paddleslim中提供用初始的模型结构来构造搜索空间的话，仅需要指定搜索空间名字即可。例如：如果使用原本的MobileNetV2的搜索空间进行搜索的话，传入SANAS中的configs直接指定为[('MobileNetV2Space')]。
2. 使用paddleslim中提供的block搜索空间构造搜索空间：<br>
  2.1 使用`input_size`, `output_size`和`block_num`来构造搜索空间。例如：传入SANAS的configs可以指定为[('MobileNetV2BlockSpace', {'input_size': 224, 'output_size': 32, 'block_num': 10})]。<br>
  2.2 使用`block_mask`构造搜索空间。例如：传入SANAS的configs可以指定为[('MobileNetV2BlockSpace', {'block_mask': [0, 1, 1, 1, 1, 0, 1, 0]})]。


## 自定义搜索空间(search space)

自定义搜索空间类需要继承搜索空间基类并重写以下几部分：<br>
&emsp; 1. 初始化的tokens(`init_tokens`函数)，可以设置为自己想要的tokens列表, tokens列表中的每个数字指的是当前数字在相应的搜索列表中的索引。例如本示例中若tokens=[0, 3, 5]，则代表当前模型结构搜索到的通道数为[8, 40, 128]。<br>
&emsp; 2. tokens中每个数字的搜索列表长度(`range_table`函数)，tokens中每个token的索引范围。<br>
&emsp; 3. 根据tokens产生模型结构(`token2arch`函数)，根据搜索到的tokens列表产生模型结构。 <br>

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

    ### 定义token的index的取值范围
    def range_table(self):
        return [len(self.filter_num)] * 3 * len(self.block_mask)

    ### 把token转换成模型结构
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

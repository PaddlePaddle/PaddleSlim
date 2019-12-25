## SANAS
SANAS（Simulated Annealing Neural Architecture Search）是基于模拟退火算法进行模型结构搜索的算法，一般用于离散搜索任务。

paddleslim.nas.SANAS(configs, server_addr, init_temperature, reduce_rate, search_steps, save_checkpoint, load_checkpoint, is_server)[源代码](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/sa_nas.py#L36)

**参数：**
- **configs(list<tuple>)** 搜索空间配置列表，格式是`[(key, {input_size, output_size, block_num, block_mask})]`或者`[(key)]`（MobileNetV2、MobilenetV1和ResNet的搜索空间使用和原本网络结构相同的搜索空间，所以仅需指定`key`即可）, `input_size` 和`output_size`表示输入和输出的特征图的大小，`block_num`是指搜索网络中的block数量，`block_mask`是一组由0和1组成的列表，0代表不进行下采样的block，1代表下采样的block。 更多paddleslim提供的搜索空间配置可以参考。
- **server_addr(tuple)** SANAS的地址，包括server的ip地址和端口号，如果ip地址为None或者为""的话则默认使用本机ip。默认：（"", 8881）。
- **init_temperature(float)** 基于模拟退火进行搜索的初始温度。默认：100。
- **reduce_rate(float)** 基于模拟退火进行搜索的衰减率。默认：0.85。
- **search_steps(int)** 搜索过程迭代的次数。默认：300。
- **save_checkpoint(str|None)** 保存checkpoint的文件目录，如果设置为None的话则不保存checkpoint。默认：`./nas_checkpoint`。
- **load_checkpoint(str|None)** 加载checkpoint的文件目录，如果设置为None的话则不加载checkpoint。默认：None。
- **is_server(bool)** 当前实例是否要启动一个server。默认：True。

**返回：**
一个SANAS类的实例

**示例代码：**
```
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(config=config)
```


paddlesim.nas.SANAS.tokens2arch(tokens)
通过一组token得到实际的模型结构，一般用来把搜索到最优的token转换为模型结构用来做最后的训练。

**参数：**
- **tokens(list):** 一组token。

**返回：**
根据传入的token得到一个模型结构实例。

**示例代码：**
```
import paddle.fluid as fluid
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
archs = sanas.token2arch(tokens)
for arch in archs:
    output = arch(input)
    input = output
```

paddleslim.nas.SANAS.next_archs():
获取下一组模型结构。

**返回：**
返回模型结构实例的列表，形式为list。

**示例代码：**
```
import paddle.fluid as fluid
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
archs = sanas.next_archs()
for arch in archs:
    output = arch(input)
    input = output
```


paddleslim.nas.SANAS.reward(score):
把当前模型结构的得分情况回传。

**参数：**
**score<float>:** 当前模型的得分，分数越大越好。

**返回：**
模型结构更新成功或者失败，成功则返回`True`，失败则返回`False`。

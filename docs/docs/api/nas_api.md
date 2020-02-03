## 搜索空间参数的配置
通过参数配置搜索空间。更多搜索空间的使用可以参考[search_space](../search_space.md)

**参数：**

- **input_size(int|None)**：- `input_size`表示输入feature map的大小。`input_size`和`output_size`用来计算整个模型结构中下采样次数。
- **output_size(int|None)**：- `output_size`表示输出feature map的大小。`input_size`和`output_size`用来计算整个模型结构中下采样次数。
- **block_num(int|None)**：- `block_num`表示搜索空间中block的数量。
- **block_mask(list|None)**：- `block_mask`是一组由0、1组成的列表，0表示当前block是normal block，1表示当前block是reduction block。reduction block表示经过这个block之后的feature map大小下降为之前的一半，normal block表示经过这个block之后feature map大小不变。如果设置了`block_mask`，则主要以`block_mask`为主要配置，`input_size`，`output_size`和`block_num`三种配置是无效的。

## SANAS

paddleslim.nas.SANAS(configs, server_addr=("", 8881), init_temperature=None, reduce_rate=0.85, init_tokens=None, search_steps=300, save_checkpoint='./nas_checkpoint', load_checkpoint=None, is_server=True)[源代码](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/sa_nas.py#L36)
: SANAS（Simulated Annealing Neural Architecture Search）是基于模拟退火算法进行模型结构搜索的算法，一般用于离散搜索任务。

**参数：**

- **configs(list<tuple>)** - 搜索空间配置列表，格式是`[(key, {input_size, output_size, block_num, block_mask})]`或者`[(key)]`（MobileNetV2、MobilenetV1和ResNet的搜索空间使用和原本网络结构相同的搜索空间，所以仅需指定`key`即可）, `input_size` 和`output_size`表示输入和输出的特征图的大小，`block_num`是指搜索网络中的block数量，`block_mask`是一组由0和1组成的列表，0代表不进行下采样的block，1代表下采样的block。 更多paddleslim提供的搜索空间配置可以参考[Search Space](../search_space.md)。
- **server_addr(tuple)** - 服务器端的地址，包括server的ip地址和端口号，如果`is_server = True`而且ip地址为None或者为""的话则默认使用本机ip。默认：（"", 8881）。
- **init_temperature(float)** - 基于模拟退火进行搜索的初始温度。如果`init_temperature`为None而且`init_tokens`为None，则默认初始温度为10.0，如果`init_temperature`为None且`init_tokens`不为None，则默认初始温度为1.0。详细的温度设置可以参考下面的Note。默认：None。
- **reduce_rate(float)** - 基于模拟退火进行搜索的衰减率。详细的退火率设置可以参考下面的Note。默认：0.85。
- **init_tokens(list|None)** - 初始化token，若`init_tokens`为None，则SA算法随机生成初始化tokens。默认：None。
- **search_steps(int)** - 搜索过程迭代的次数。默认：300。
- **save_checkpoint(str|None)** - 保存checkpoint的文件目录，如果设置为None的话则不保存checkpoint。默认：`./nas_checkpoint`。
- **load_checkpoint(str|None)** - 加载checkpoint的文件目录，如果设置为None的话则不加载checkpoint。默认：None。
- **is_server(bool)** - 当前实例是否要启动一个server。默认：True。

**返回：**
一个SANAS类的实例

**示例代码：**
```python
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
```

!!! note "Note"
  - 初始化温度和退火率的意义: <br>
    - SA算法内部会保存一个基础token（初始化token是第一个基础token，可以自己传入也可以随机生成）和基础score（初始化score为-1），下一个token会在当前SA算法保存的token的基础上产生。在SA的搜索过程中，如果本轮的token训练得到的score大于SA算法中保存的score，则本轮的token一定会被SA算法接收保存为下一轮token产生的基础token。<br>
    - 初始温度越高表示SA算法当前处的阶段越不稳定，本轮的token训练得到的score小于SA算法中保存的score的话，本轮的token和score被SA算法接收的可能性越大。<br>
    - 初始温度越低表示SA算法当前处的阶段越稳定，本轮的token训练得到的score小于SA算法中保存的score的话，本轮的token和score被SA算法接收的可能性越小。<br>
    - 退火率越大，表示SA算法收敛的越慢，即SA算法越慢到稳定阶段。<br>
    - 退火率越低，表示SA算法收敛的越快，即SA算法越快到稳定阶段。<br>

  - 初始化温度和退火率的设置: <br>
    - 如果原本就有一个较好的初始化token，想要基于这个较好的token来进行搜索的话，SA算法可以处于一个较为稳定的状态进行搜索r这种情况下初始温度可以设置的低一些，例如设置为1.0，退火率设置的大一些，例如设置为0.85。如果想要基于这个较好的token利用贪心算法进行搜索，即只有当本轮token训练得到的score大于SA算法中保存的score，SA算法才接收本轮token，则退火率可设置为一个极小的数字，例如设置为0.85 ** 10。<br>
    - 初始化token如果是随机生成的话，代表初始化token是一个比较差的token，SA算法可以处于一种不稳定的阶段进行搜索，尽可能的随机探索所有可能得token，从而找到一个较好的token。初始温度可以设置的高一些，例如设置为1000，退火率相对设置的小一些。


paddleslim.nas.SANAS.next_archs()
: 获取下一组模型结构。

**返回：**
返回模型结构实例的列表，形式为list。

**示例代码：**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
archs = sanas.next_archs()
for arch in archs:
    output = arch(input)
    input = output
print(output)
```

paddleslim.nas.SANAS.reward(score)
: 把当前模型结构的得分情况回传。

**参数：**

- **score<float>:** - 当前模型的得分，分数越大越好。

**返回：**
模型结构更新成功或者失败，成功则返回`True`，失败则返回`False`。

**示例代码：**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
archs = sanas.next_archs()

### 假设网络计算出来的score是1，实际代码中使用时需要返回真实score。
score=float(1.0)
sanas.reward(float(score))
```


paddlesim.nas.SANAS.tokens2arch(tokens)
: 通过一组tokens得到实际的模型结构，一般用来把搜索到最优的token转换为模型结构用来做最后的训练。tokens的形式是一个列表，tokens映射到搜索空间转换成相应的网络结构，一组tokens对应唯一的一个网络结构。

**参数：**

- **tokens(list):** - 一组tokens。tokens的长度和取值范围取决于搜索空间。

**返回：**
根据传入的token得到一个模型结构实例。

**示例代码：**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
tokens = ([0] * 25)
archs = sanas.tokens2arch(tokens)[0]
print(archs(input))
```

paddleslim.nas.SANAS.current_info()
: 返回当前token和搜索过程中最好的token和reward。

**返回：**
搜索过程中最好的token，reward和当前训练的token，形式为dict。

**示例代码：**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
print(sanas.current_info())
```

# paddleslim.nas API文档

## SANAS API文档

### paddleslim.nas.SANAS(configs, server_addr, init_temperature, reduce_rate, search_steps, save_checkpoint, load_checkpoint, is_server)
初始化一个sanas实例。

**参数：**
- **configs(list<tuple>): 搜索空间配置列表，格式是[(key, {input_size, output_size, block_num, block_mask})], `input_size` 和`output_size`表示输入和输出的特征图的大小，`block_num`是指搜索网络中的block数量，`block_mask`是一组由0和1组成的列表，0代表不进行下采样的block，1代表下采样的block。 更多paddleslim提供的搜索空间配置可以参考。
- **server_addr(tuple): SANAS的地址，包括server的ip地址和端口号，如果ip地址为None或者为""的话则默认使用本机ip。默认：（"", 8881）。
- **init_temperature(float): 基于模拟退火进行搜索的初始温度。默认：100。
- **reduce_rate(float): 基于模拟退火进行搜索的衰减率。默认：0.85。
- **search_steps(int): 搜索过程迭代的次数。默认：300。
- **save_checkpoint(str|None): 保存checkpoint的文件目录，如果设置为None的话则不保存checkpoint。默认：nas_checkpoint。
- **load_checkpoint(str|None): 加载checkpoint的文件目录，如果设置为None的话则不加载checkpoint。默认：None。
- **is_server(bool): 当前实例是否要启动一个server。默认：True。

### paddleslim.nas.SANAS.tokens2arch(tokens)
通过一组token得到实际的模型结构，一般用来把搜索到最优的token转换为模型结构用来做最后的训练。

**参数：**
- **tokens(list): 搜索出来的token。

**返回**
返回一个模型模型结构实例。

**返回类型**
function

### paddleslim.nas.SANAS.next_archs():
获取下一组模型结构。

**返回**
返回模型结构实例的列表，形式为list<model_arch>。

### paddleslim.nas.SANAS.reward(score):
把当前模型结构的得分情况回传给server，server根据得分判断是否是最优得分。

**参数：**
score<float>: 当前模型的得分，分数越大越好。

**返回**
模型结构更新成功或者失败，成功则返回`True`，失败则返回`False`。

**返回类型**
bool类型


**代码示例**
```python
import paddleslim.nas.SANAS as SANAS

# 搜索空间配置
config=[('MobileNetV2Space')] 

# 实例化SANAS
sa_nas = SANAS(config, server_addr=("", 8887), init_temperature=10.24, reduce_rate=0.85, search_steps=100, is_server=True)

input = fluid.data(name='input', shape=[None, 1, 32, 32], dtype='float32')
label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
for step in range(100):
    archs = sa_nas.next_archs()
    for arch in archs:
        input = arch(input)

    score = fluid.layer.
    sa_nas.reward(score)

```

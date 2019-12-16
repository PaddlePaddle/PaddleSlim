# paddleslim.nas API文档

## SANAS API文档

## class SANAS
SANAS（Simulated Annealing Neural Architecture Search）是基于模拟退火算法进行模型结构搜索的算法，一般用于离散搜索任务。

---

>paddleslim.nas.SANAS(configs, server_addr, init_temperature, reduce_rate, search_steps, save_checkpoint, load_checkpoint, is_server)

**参数：**
- **configs(list<tuple>):** 搜索空间配置列表，格式是`[(key, {input_size, output_size, block_num, block_mask})]`或者`[(key)]`（MobileNetV2、MobilenetV1和ResNet的搜索空间使用和原本网络结构相同的搜索空间，所以仅需指定`key`即可）, `input_size` 和`output_size`表示输入和输出的特征图的大小，`block_num`是指搜索网络中的block数量，`block_mask`是一组由0和1组成的列表，0代表不进行下采样的block，1代表下采样的block。 更多paddleslim提供的搜索空间配置可以参考。
- **server_addr(tuple):** SANAS的地址，包括server的ip地址和端口号，如果ip地址为None或者为""的话则默认使用本机ip。默认：（"", 8881）。
- **init_temperature(float):** 基于模拟退火进行搜索的初始温度。默认：100。
- **reduce_rate(float):** 基于模拟退火进行搜索的衰减率。默认：0.85。
- **search_steps(int):** 搜索过程迭代的次数。默认：300。
- **save_checkpoint(str|None):** 保存checkpoint的文件目录，如果设置为None的话则不保存checkpoint。默认：`./nas_checkpoint`。
- **load_checkpoint(str|None):** 加载checkpoint的文件目录，如果设置为None的话则不加载checkpoint。默认：None。
- **is_server(bool):** 当前实例是否要启动一个server。默认：True。

**返回：** 
一个SANAS类的实例

**示例代码：**
```
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(config=config)
```

---

>tokens2arch(tokens)
通过一组token得到实际的模型结构，一般用来把搜索到最优的token转换为模型结构用来做最后的训练。

**参数：**
- **tokens(list):** 一组token。

**返回**
返回一个模型结构实例。

**示例代码：**
```
import paddle.fluid as fluid
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
archs = sanas.token2arch(tokens)
for arch in archs:
    output = arch(input)
    input = output
```
---

>next_archs():
获取下一组模型结构。

**返回**
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

---

>reward(score):
把当前模型结构的得分情况回传。

**参数：**
**score<float>:** 当前模型的得分，分数越大越好。

**返回**
模型结构更新成功或者失败，成功则返回`True`，失败则返回`False`。


**代码示例**
```python
import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.nas import SANAS
from paddleslim.analysis import flops

max_flops = 321208544
batch_size = 256

# 搜索空间配置
config=[('MobileNetV2Space')] 

# 实例化SANAS
sa_nas = SANAS(config, server_addr=("", 8887), init_temperature=10.24, reduce_rate=0.85, search_steps=100, is_server=True)

for step in range(100):
    archs = sa_nas.next_archs()
    train_program = fluid.Program()
    test_program = fluid.Program()
    startup_program = fluid.Program()
    ### 构造训练program
    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='image', shape=[None, 3, 32, 32], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')

        for arch in archs:
            output = arch(image)
        out = fluid.layers.fc(output, size=10, act="softmax") 
        softmax_out = fluid.layers.softmax(input=out, use_cudnn=False)
        cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)

        ### 构造测试program
        test_program = train_program.clone(for_test=True)
        ### 定义优化器
        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(avg_cost)


    ### 增加限制条件，如果没有则进行无限制搜索
    if flops(train_program) > max_flops:
        continue

    ### 定义代码是在cpu上运行
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    ### 定义训练输入数据
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10(cycle=False), buf_size=1024),
        batch_size=batch_size,
        drop_last=True)

    ### 定义预测输入数据
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(cycle=False),
        batch_size=batch_size,
        drop_last=False)
    train_feeder = fluid.DataFeeder([image, label], place, program=train_program)
    test_feeder = fluid.DataFeeder([image, label], place, program=test_program)


    ### 开始训练，每个搜索结果训练5个epoch
    for epoch_id in range(5):
        for batch_id, data in enumerate(train_reader()):
            fetches = [avg_cost.name]
            outs = exe.run(train_program,
                           feed=train_feeder.feed(data),
                           fetch_list=fetches)[0]
            if batch_id % 10 == 0:
                print('TRAIN: steps: {}, epoch: {}, batch: {}, cost: {}'.format(step, epoch_id, batch_id, outs[0]))

    ### 开始预测，得到最终的测试结果作为score回传给sa_nas
    reward = []
    for batch_id, data in enumerate(test_reader()):
        test_fetches = [
            avg_cost.name, acc_top1.name
        ]
        batch_reward = exe.run(test_program,
                               feed=test_feeder.feed(data),
                               fetch_list=test_fetches)
        reward_avg = np.mean(np.array(batch_reward), axis=1)
        reward.append(reward_avg)

        print('TEST: step: {}, batch: {}, avg_cost: {}, acc_top1: {}'.
            format(step, batch_id, batch_reward[0],batch_reward[1]))

    finally_reward = np.mean(np.array(reward), axis=0)
    print(
        'FINAL TEST: avg_cost: {}, acc_top1: {}'.format(
            finally_reward[0], finally_reward[1]))

    ### 回传score
    sa_nas.reward(float(finally_reward[1]))

```

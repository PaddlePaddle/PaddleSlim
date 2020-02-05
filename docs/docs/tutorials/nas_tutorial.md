# 图像分类网络结构搜索-快速开始

该教程以图像分类模型MobileNetV2为例，说明如何在cifar10数据集上快速使用[网络结构搜索接口](../api/nas_api.md)。
该示例包含以下步骤：

1. 导入依赖
2. 初始化SANAS搜索实例
3. 构建网络
4. 启动搜索实验

以下章节依次介绍每个步骤的内容。

## 1. 导入依赖
请确认已正确安装Paddle，导入需要的依赖包。
```python
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 初始化SANAS搜索实例
```python
sanas = slim.nas.SANAS(configs=[('MobileNetV2Space')])
```

## 3. 构建网络
根据传入的网络结构构造训练program和测试program。
```python
def build_program(archs):
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        output = archs(data)
        output = fluid.layers.fc(input=output, size=10)

        softmax_out = fluid.layers.softmax(input=output, use_cudnn=False)
        cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)
        test_program = fluid.default_main_program().clone(for_test=True)
            
        optimizer = fluid.optimizer.Adam(learning_rate=0.1)
        optimizer.minimize(avg_cost)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
    return exe, train_program, test_program, (data, label), avg_cost, acc_top1, acc_top5
```

## 4. 启动搜索实验
获取每一轮的模型结构并开始训练。该教程中使用FLOPs作为约束条件，搜索实验一共搜索3个step，表示搜索到3个满足条件的模型结构进行训练，每搜索到一个网络结构训练7个epoch。
```python
for step in range(3):
    archs = sanas.next_archs()[0]
    exe, train_program, test_progarm, inputs, avg_cost, acc_top1, acc_top5 = build_program(archs)

    current_flops = slim.analysis.flops(train_program)
    if current_flops > 321208544:
        continue
    
    train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.cifar.train10(cycle=False),                          buf_size=1024),batch_size=256)
    train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
    test_reader = paddle.batch(paddle.dataset.cifar.test10(cycle=False),
               batch_size=256)
    test_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())

    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for epoch in range(7):
        for data in train_reader():
            loss, acc1, acc5 = exe.run(train_program, feed=train_feeder.feed(data), fetch_list = outputs)
            print("TRAIN: loss: {}, acc1: {}, acc5:{}".format(loss, acc1, acc5))

    reward = []
    for data in test_reader():
        batch_reward = exe.run(test_program, feed=test_feeder.feed(data), fetch_list = outputs)
        reward_avg = np.mean(np.array(batch_reward), axis=1)
        reward.append(reward_avg)
        print("TEST: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))
    finally_reward = np.mean(np.array(reward), axis=0)
    print("FINAL TEST: avg_cost: {}, acc1: {}, acc5: {}".format(finally_reward[0], finally_reward[1], finally_reward[2]))

    sanas.reward(float(finally_reward[1]))
```

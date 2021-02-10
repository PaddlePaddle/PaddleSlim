# 网络结构搜索

该教程以图像分类模型MobileNetV2为例，说明如何在cifar10数据集上快速使用[网络结构搜索接口](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/nas/nas_api.html)。
该示例包含以下步骤：

1. 导入依赖
2. 初始化SANAS搜索实例
3. 构建网络
4. 定义输入数据函数
5. 定义训练函数
6. 定义评估函数
7. 启动搜索实验  
  7.1 获取模型结构  
  7.2 构造program  
  7.3 定义输入数据  
  7.4 训练模型  
  7.5 评估模型  
  7.6 回传当前模型的得分
8. 完整示例


以下章节依次介绍每个步骤的内容。

## 1. 导入依赖
请确认已正确安装Paddle，导入需要的依赖包。
```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.static as static
import paddleslim as slim
import numpy as np
```

## 2. 初始化SANAS搜索实例
```python
sanas = slim.nas.SANAS(configs=[('MobileNetV2Space')], server_addr=("", 8337), save_checkpoint=None)
```

## 3. 构建网络
根据传入的网络结构构造训练program和测试program。
```python
paddle.enable_static()
def build_program(archs):
    train_program = static.Program()
    startup_program = static.Program()
    with static.program_guard(train_program, startup_program):
        data = static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
        label = static.data(name='label', shape=[None, 1], dtype='int64')
        gt = paddle.reshape(label, [-1, 1])
        output = archs(data)
        output = static.nn.fc(output, size=10)

        softmax_out = F.softmax(output)
        cost = F.cross_entropy(softmax_out, label=label)
        avg_cost = paddle.mean(cost)
        acc_top1 = paddle.metric.accuracy(input=softmax_out, label=gt, k=1)
        acc_top5 = paddle.metric.accuracy(input=softmax_out, label=gt, k=5)
        test_program = static.default_main_program().clone(for_test=True)

        optimizer = paddle.optimizer.Adam(learning_rate=0.1)
        optimizer.minimize(avg_cost)

        place = paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(startup_program)
    return exe, train_program, test_program, (data, label), avg_cost, acc_top1, acc_top5
```

## 4. 定义输入数据函数
为了快速执行该示例，我们使用的数据集为CIFAR10，Paddle框架的`paddle.vision.datasets.Cifar10`包定义了CIFAR10数据的下载和读取。 代码如下：

```python
import paddle.vision.transforms as T

def input_data(image, label):
    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    train_dataset = paddle.vision.datasets.Cifar10(mode="train", transform=transform, backend='cv2')
    train_loader = paddle.io.DataLoader(train_dataset,
                    places=paddle.CPUPlace(),
                    feed_list=[image, label],
                    drop_last=True,
                    batch_size=64,
                    return_list=False,
                    shuffle=True)
    eval_dataset = paddle.vision.datasets.Cifar10(mode="test", transform=transform, backend='cv2')
    eval_loader = paddle.io.DataLoader(eval_dataset,
                    places=paddle.CPUPlace(),
                    feed_list=[image, label],
                    drop_last=False,
                    batch_size=64,
                    return_list=False,
                    shuffle=False)
    return train_loader, eval_loader
```

## 5. 定义训练函数
根据训练program和训练数据进行训练。
```python
def start_train(program, data_loader):
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_loader():
        batch_reward = exe.run(program, feed=data, fetch_list = outputs)
        print("TRAIN: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))
```

## 6. 定义评估函数
根据评估program和评估数据进行评估。
```python
def start_eval(program, data_loader):
    reward = []
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_loader():
        batch_reward = exe.run(program, feed=data, fetch_list = outputs)
        reward_avg = np.mean(np.array(batch_reward), axis=1)
        reward.append(reward_avg)
        print("TEST: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))
    finally_reward = np.mean(np.array(reward), axis=0)
    print("FINAL TEST: avg_cost: {}, acc1: {}, acc5: {}".format(finally_reward[0], finally_reward[1], finally_reward[2]))
    return finally_reward
```

## 7. 启动搜索实验
以下步骤拆解说明了如何获得当前模型结构以及获得当前模型结构之后应该有的步骤，如果想要看如何启动搜索实验的完整示例可以看步骤9。

### 7.1 获取模型结构
调用`next_archs()`函数获取到下一个模型结构。
```python
archs = sanas.next_archs()[0]
```

### 7.2 构造program
调用步骤3中的函数，根据4.1中的模型结构构造相应的program。
```python
exe, train_program, eval_program, (image, label), avg_cost, acc_top1, acc_top5 = build_program(archs)
```

### 7.3 定义输入数据
```python
train_loader, eval_loader = input_data(image, label)
```

### 7.4 训练模型
根据上面得到的训练program和评估数据启动训练。
```python
start_train(train_program, train_loader)
```
### 7.5 评估模型
根据上面得到的评估program和评估数据启动评估。
```python
finally_reward = start_eval(eval_program, eval_loader)
```
### 7.6 回传当前模型的得分
```
sanas.reward(float(finally_reward[1]))
```

## 8. 完整示例
以下是一个完整的搜索实验示例，示例中使用FLOPs作为约束条件，搜索实验一共搜索3个step，表示搜索到3个满足条件的模型结构进行训练，每搜索到一个网络结构训练7个epoch。
```python
for step in range(3):
    archs = sanas.next_archs()[0]
    exe, train_program, eval_program, inputs, avg_cost, acc_top1, acc_top5 = build_program(archs)
    train_loader, eval_loader = input_data(inputs)

    current_flops = slim.analysis.flops(train_program)
    if current_flops > 321208544:
        continue

    for epoch in range(7):
        start_train(train_program, train_loader)

    finally_reward = start_eval(eval_program, eval_loader)

    sanas.reward(float(finally_reward[1]))
```

# Nerual Architecture Search for Image Classification

This tutorial shows how to use [API](https://paddleslim.readthedocs.io/en/latest/api_en/paddleslim.nas.html) about SANAS in PaddleSlim. We start experiment based on MobileNetV2 as example. The tutorial contains follow section.

1. necessary imports
2. initial SANAS instance
3. define function about building program
4. define function about input data
5. define function about training
6. define funciton about evaluation
7. start search
  7.1 fetch model architecture
  7.2 build program
  7.3 define input data
  7.4 train model
  7.5 evaluate model
  7.6 reture score
8. full example


The following chapter describes each steps in order.

## 1. import dependency
Please make sure that you haved installed Paddle correctly, then do the necessary imports.
```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.static as static
import paddleslim as slim
import numpy as np
```

## 2. initial SANAS instance

Please set a unused port when build instance of SANAS.
```python
port = np.random.randint(8337, 8773)
sanas = slim.nas.SANAS(configs=[('MobileNetV2Space')], server_addr=("", port), save_checkpoint=None)
```

## 3. define function about building program
Build program about training and evaluation according to the model architecture.
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
        cost = F.cross_entropy(softmax_out, label=gt)
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

## 4. define function about input data
The dataset we used is cifar10, and `paddle.dataset.cifar` in Paddle including the download and pre-read about cifar.
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

## 5. define function about training
Start training.
```python
def start_train(program, data_loader):
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_loader():
        batch_reward = exe.run(program, feed=data, fetch_list = outputs)
        print("TRAIN: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))
```

## 6. define funciton about evaluation
Start evaluating.
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

## 7. start search
The following steps describes how to get current model architecture and what need to do after get the model architecture. If you want to start a full example directly, please jump to Step 9.

### 7.1 fetch model architecture
Get Next model architecture by `next_archs()`.
```python
archs = sanas.next_archs()[0]
```

### 7.2 build program
Get program according to the function in Step3 and model architecture from Step 7.1.
```python
paddle.enable_static()
exe, train_program, eval_program, (image, label), avg_cost, acc_top1, acc_top5 = build_program(archs)
```

### 7.3 define input data
```python
train_loader, eval_loader = input_data(image, label)
```

### 7.4 train model
Start training according to train program and data.
```python
start_train(train_program, train_loader)
```
### 7.5 evaluate model
Start evaluation according to evaluation program and data.
```python
finally_reward = start_eval(eval_program, eval_loader)
```
### 7.6 reture score
```
sanas.reward(float(finally_reward[1]))
```

## 8. full example
The following is a full example about neural architecture search, it uses FLOPs as constraint and includes 3 steps, it means train 3 model architectures which is satisfied constraint, and train 7 epoch for each model architecture.
```python
for step in range(3):
    archs = sanas.next_archs()[0]
    exe, train_program, eval_program, (images,label), avg_cost, acc_top1, acc_top5 = build_program(archs)
    train_loader, eval_loader = input_data(images, label)

    current_flops = slim.analysis.flops(train_program)
    if current_flops > 321208544:
        continue

    for epoch in range(7):
        start_train(train_program, train_loader)

    finally_reward = start_eval(eval_program, eval_loader)

    sanas.reward(float(finally_reward[1]))
```

# Nerual Architecture Search for Image Classification

This tutorial shows how to use [API](../api/nas_api.md) about SANAS in PaddleSlim. We start experiment based on MobileNetV2 as example. The tutorial contains follow section.

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
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. initial SANAS instance
```python
sanas = slim.nas.SANAS(configs=[('MobileNetV2Space')], server_addr=("", 8337), save_checkpoint=None)
```

## 3. define function about building program
Build program about training and evaluation according to the model architecture.
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

## 4. define function about input data
The dataset we used is cifar10, and `paddle.dataset.cifar` in Paddle including the download and pre-read about cifar.
```python
def input_data(inputs):
    train_reader = paddle.fluid.io.batch(paddle.reader.shuffle(paddle.dataset.cifar.train10(cycle=False), buf_size=1024),batch_size=256)
    train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
    eval_reader = paddle.fluid.io.batch(paddle.dataset.cifar.test10(cycle=False), batch_size=256)
    eval_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
    return train_reader, train_feeder, eval_reader, eval_feeder
```

## 5. define function about training
Start training.
```python
def start_train(program, data_reader, data_feeder):
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_reader():
        batch_reward = exe.run(program, feed=data_feeder.feed(data), fetch_list = outputs)
        print("TRAIN: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))
```

## 6. define funciton about evaluation
Start evaluating.
```python
def start_eval(program, data_reader, data_feeder):
    reward = []
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_reader():
        batch_reward = exe.run(program, feed=data_feeder.feed(data), fetch_list = outputs)
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
exe, train_program, eval_program, inputs, avg_cost, acc_top1, acc_top5 = build_program(archs)
```

### 7.3 define input data
```python
train_reader, train_feeder, eval_reader, eval_feeder = input_data(inputs)
```

### 7.4 train model
Start training according to train program and data.
```python
start_train(train_program, train_reader, train_feeder)
```
### 7.5 evaluate model
Start evaluation according to evaluation program and data.
```python
finally_reward = start_eval(eval_program, eval_reader, eval_feeder)
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
    exe, train_program, eval_progarm, inputs, avg_cost, acc_top1, acc_top5 = build_program(archs)
    train_reader, train_feeder, eval_reader, eval_feeder = input_data(inputs)

    current_flops = slim.analysis.flops(train_program)
    if current_flops > 321208544:
        continue

    for epoch in range(7):
        start_train(train_program, train_reader, train_feeder)

    finally_reward = start_eval(eval_program, eval_reader, eval_feeder)

    sanas.reward(float(finally_reward[1]))
```

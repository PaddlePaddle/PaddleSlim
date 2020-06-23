# SANAS进阶版实验教程-压缩DARTS产出模型

## 收益情况
利用DARTS搜索出来的最终模型结构（以下简称为DARTS_model）构造相应的搜索空间，根据PaddleSlim提供的SANAS搜索方法进行搜索实验，最终得到的模型结构（以下简称为DARTS_SA）相比DARTS_model的精度提升<font color=green>0.141%</font>，模型大小下降<font color=green>11.2%</font>。

## 搜索教程
本教程展示了如何在DARTS_model基础上利用SANAS进行搜索实验，并得到DARTS_SA的结果。

本教程包含以下步骤：
1. 构造搜索空间
2. 导入依赖包并定义全局变量
3. 初始化SANAS实例
4. 定义计算模型参数量的函数
5. 定义网络输入数据的函数
6. 定义造program的函数
7. 定义训练函数
8. 定义预测函数
9. 启动搜索  
  9.1 获取下一个模型结构  
  9.2 构造相应的训练和预测program  
  9.3 添加搜索限制  
  9.4 定义环境  
  9.5 定义输入数据  
  9.6 启动训练和评估  
  9.7 回传当前模型的得分reward
10. 利用demo下的脚本启动搜索
11. 利用demo下的脚本启动最终实验

### 1. 构造搜索空间
进行搜索实验之前，首先需要根据DARTS_model的模型特点构造相应的搜索空间，本次实验仅会对DARTS_model的通道数进行搜索，搜索的目的是得到一个精度更高并且模型参数更少的模型。
定义如下搜索空间：
- 通道数`filter_num`: 定义了每个卷积操作的通道数变化区间。取值区间为：`[4, 8, 12, 16, 20, 36, 54, 72, 90, 108, 144, 180, 216, 252]`

按照通道数来区分DARTS_model中block的话，则DARTS_model中共有3个block，第一个block仅包含6个normal cell，之后的两个block每个block都包含和一个reduction cell和6个normal cell，共有20个cell。在构造搜索空间的时候我们定义每个cell中的所有卷积操作都使用相同的通道数，共有20位token。

完整的搜索空间可以参考[基于DARTS_model的搜索空间](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/search_space/darts_space.py)

### 2. 引入依赖包并定义全局变量
```python
import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.nas import SANAS

BATCH_SIZE=96
SERVER_ADDRESS = ""
PORT = 8377
SEARCH_STEPS = 300
RETAIN_EPOCH=30
MAX_PARAMS=3.77
IMAGE_SHAPE=[3, 32, 32]
AUXILIARY = True
AUXILIARY_WEIGHT= 0.4
TRAINSET_NUM = 50000
LR = 0.025
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0003
DROP_PATH_PROBILITY = 0.2
```

### 3. 初始化SANAS实例
首先需要初始化SANAS示例。
```python
config = [('DartsSpace')]
sa_nas = SANAS(config, server_addr=(SERVER_ADDRESS, PORT), search_steps=SEARCH_STEPS, is_server=True)
```

### 4. 定义计算模型参数量的函数
根据输入的program计算当前模型中的参数量。本教程使用模型参数量作为搜索的限制条件。
```python
def count_parameters_in_MB(all_params, prefix='model'):
    parameters_number = 0
    for param in all_params:
        if param.name.startswith(
                prefix) and param.trainable and 'aux' not in param.name:
            parameters_number += np.prod(param.shape)
    return parameters_number / 1e6
```

### 5. 定义网络输入数据的函数
根据输入图片的尺寸定义网络中的输入，其中包括图片输入、标签输入和在训练过程中需要随机丢弃单元的比例和掩膜。
```python
def create_data_loader(IMAGE_SHAPE, is_train):
    image = fluid.data(
        name="image", shape=[None] + IMAGE_SHAPE, dtype="float32")
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[image, label],
        capacity=64,
        use_double_buffer=True,
        iterable=True)
    drop_path_prob = ''
    drop_path_mask = ''
    if is_train:
        drop_path_prob = fluid.data(
            name="drop_path_prob", shape=[BATCH_SIZE, 1], dtype="float32")
        drop_path_mask = fluid.data(
            name="drop_path_mask",
            shape=[BATCH_SIZE, 20, 4, 2],
            dtype="float32")

    return data_loader, image, label, drop_path_prob, drop_path_mask
```

### 6. 定义构造program的函数
根据输入的模型结构、输入图片尺寸和当前program是否是训练模式构造program。
```python
def build_program(main_program, startup_program, IMAGE_SHAPE, archs, is_train):
    with fluid.program_guard(main_program, startup_program):
        data_loader, data, label, drop_path_prob, drop_path_mask = create_data_loader(
            IMAGE_SHAPE, is_train)
        logits, logits_aux = archs(data, drop_path_prob, drop_path_mask,
                                   is_train, 10)
        top1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        top5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            fluid.layers.softmax_with_cross_entropy(logits, label))

        if is_train:
            if AUXILIARY:
                loss_aux = fluid.layers.reduce_mean(
                    fluid.layers.softmax_with_cross_entropy(logits_aux, label))
                loss = loss + AUXILIARY_WEIGHT * loss_aux
            step_per_epoch = int(TRAINSET_NUM / BATCH_SIZE)
            learning_rate = fluid.layers.cosine_decay(LR, step_per_epoch, RETAIN_EPOCH)
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
            optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate,
                MOMENTUM,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    WEIGHT_DECAY))
            optimizer.minimize(loss)
            outs = [loss, top1, top5, learning_rate]
        else:
            outs = [loss, top1, top5]
    return outs, data_loader

```

### 7. 定义训练函数
```python
def train(main_prog, exe, epoch_id, train_loader, fetch_list):
    loss = []
    top1 = []
    top5 = []
    for step_id, data in enumerate(train_loader()):
        devices_num = len(data)
        if DROP_PATH_PROBILITY > 0:
            feed = []
            for device_id in range(devices_num):
                image = data[device_id]['image']
                label = data[device_id]['label']
                drop_path_prob = np.array(
                    [[DROP_PATH_PROBILITY * epoch_id / RETAIN_EPOCH]
                     for i in range(BATCH_SIZE)]).astype(np.float32)
                drop_path_mask = 1 - np.random.binomial(
                    1, drop_path_prob[0],
                    size=[BATCH_SIZE, 20, 4, 2]).astype(np.float32)
                feed.append({
                    "image": image,
                    "label": label,
                    "drop_path_prob": drop_path_prob,
                    "drop_path_mask": drop_path_mask
                })
        else:
            feed = data
        loss_v, top1_v, top5_v, lr = exe.run(
            main_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.append(loss_v)
        top1.append(top1_v)
        top5.append(top5_v)
        if step_id % 10 == 0:
            print(
                "Train Epoch {}, Step {}, Lr {:.8f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, lr[0], np.mean(loss), np.mean(top1), np.mean(top5)))
    return np.mean(top1)
```

### 8. 定义预测函数
```python
def valid(main_prog, exe, epoch_id, valid_loader, fetch_list):
    loss = []
    top1 = []
    top5 = []
    for step_id, data in enumerate(valid_loader()):
        loss_v, top1_v, top5_v = exe.run(
            main_prog, feed=data, fetch_list=[v.name for v in fetch_list])
        loss.append(loss_v)
        top1.append(top1_v)
        top5.append(top5_v)
        if step_id % 10 == 0:
            print(
                "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, np.mean(loss), np.mean(top1), np.mean(top5)))
    return np.mean(top1)
```

### 9. 启动搜索实验
以下步骤拆解说明了如何获得当前模型结构以及获得当前模型结构之后应该有的步骤。

#### 9.1 获取下一个模型结构
根据上面的SANAS实例中的函数获取下一个模型结构。
```python
archs = sa_nas.next_archs()[0]
```

#### 9.2 构造训练和预测program
根据上一步中获得的模型结构分别构造训练program和预测program。
```python
train_program = fluid.Program()
test_program = fluid.Program()
startup_program = fluid.Program()
train_fetch_list, train_loader = build_program(train_program, startup_program, IMAGE_SHAPE, archs, is_train=True)
test_fetch_list, test_loader = build_program(test_program, startup_program, IMAGE_SHAPE, archs, is_train=False)
test_program = test_program.clone(for_test=True)
```

#### 9.3 添加搜索限制
本教程以模型参数量为限制条件。首先计算一下当前program的参数量，如果超出限制条件，则终止本次模型结构的训练，获取下一个模型结构。
```python
current_params = count_parameters_in_MB(
    train_program.global_block().all_parameters(), 'cifar10')
```

#### 9.4 定义环境
定义数据和模型的环境并初始化参数。
```python
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)
```

#### 9.5 定义输入数据
由于本示例中对cifar10中的图片进行了一些额外的预处理操作，和[快速开始](https://paddlepaddle.github.io/PaddleSlim/quick_start/nas_tutorial.html)示例中的reader不同，所以需要自定义cifar10的reader，不能直接调用paddle中封装好的`paddle.dataset.cifar10`的reader。自定义cifar10的reader文件位于[demo/nas](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/nas/darts_cifar10_reader.py)中。

**注意：**本示例为了简化代码直接调用`paddle.dataset.cifar10`定义训练数据和预测数据，实际训练需要使用自定义cifar10文件中的reader。
```python
train_reader = paddle.fluid.io.batch(paddle.reader.shuffle(paddle.dataset.cifar.train10(cycle=False), buf_size=1024), batch_size=BATCH_SIZE, drop_last=True)
test_reader = paddle.fluid.io.batch(paddle.dataset.cifar.test10(cycle=False), batch_size=BATCH_SIZE, drop_last=False)
train_loader.set_sample_list_generator(train_reader, places=place)
test_loader.set_sample_list_generator(test_reader, places=place)
```

#### 9.6 启动训练和评估
```python
for epoch_id in range(RETAIN_EPOCH):
    train_top1 = train(train_program, exe, epoch_id, train_loader, train_fetch_list)
    print("TRAIN: Epoch {}, train_acc {:.6f}".format(epoch_id, train_top1))
    valid_top1 = valid(test_program, exe, epoch_id, test_loader, test_fetch_list)
    print("TEST: Epoch {}, valid_acc {:.6f}".format(epoch_id, valid_top1))
    valid_top1_list.append(valid_top1)
```

#### 9.7 回传当前模型的得分reward
本教程利用最后两个epoch的准确率均值作为最终的得分回传给SANAS。
```python
sa_nas.reward(float(valid_top1_list[-1] + valid_top1_list[-2]) / 2)
```


### 10. 利用demo下的脚本启动搜索

搜索文件位于: [darts_sanas_demo](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/nas/sanas_darts_space.py)，搜索过程中限制模型参数量为不大于3.77M。
```python
cd demo/nas/
python darts_nas.py
```

### 11. 利用demo下的脚本启动最终实验
最终实验文件位于: [darts_sanas_demo](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/nas/sanas_darts_space.py)，最终实验需要训练600epoch。以下示例输入token为`[5, 5, 0, 5, 5, 10, 7, 7, 5, 7, 7, 11, 10, 12, 10, 0, 5, 3, 10, 8]`。
```python
cd demo/nas/
python darts_nas.py --token 5 5 0 5 5 10 7 7 5 7 7 11 10 12 10 0 5 3 10 8 --retain_epoch 600
```

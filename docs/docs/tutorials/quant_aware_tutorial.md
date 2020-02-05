# 图像分类模型量化训练-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用PaddleSlim的[量化训练接口](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/docs/api/quantization_api.md)。 该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 训练模型
4. 量化
5. 训练和测试量化后的模型

## 1. 导入依赖
PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:


```python
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 构建网络
该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：

>注意：paddleslim.models下的API并非PaddleSlim常规API，是为了简化示例而封装预定义的一系列方法，比如：模型结构的定义、Program的构建等。


```python
exe, train_program, val_program, inputs, outputs = \
    slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=True)
```

## 3. 训练模型

### 3.1 定义输入数据

为了快速执行该示例，我们选取简单的MNIST数据，Paddle框架的`paddle.dataset.mnist`包定义了MNIST数据的下载和读取。
代码如下：


```python
import paddle.dataset.mnist as reader
train_reader = paddle.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.batch(
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
```

### 3.2 训练和测试
先定义训练和测试函数


```python
def train(prog):
    iter = 0
    for data in train_reader():
        acc1, acc5, loss = exe.run(prog, feed=train_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('train', acc1.mean(), acc5.mean(), loss.mean())
        iter += 1

def test(prog):
    iter = 0
    res = [[], []]
    for data in train_reader():
        acc1, acc5, loss = exe.run(prog, feed=train_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('test', acc1.mean(), acc5.mean(), loss.mean())
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result', np.array(res[0]).mean(), np.array(res[1]).mean())
```

训练分类网络


```python
train(train_program)
```

    ('train', 0.0703125, 0.4765625, 2.7081475)
    ('train', 0.9296875, 1.0, 0.2265962)
    ('train', 0.953125, 1.0, 0.18551664)
    ('train', 0.9609375, 0.9921875, 0.1773298)
    ('train', 0.953125, 1.0, 0.1571356)


测试分类网络


```python
test(val_program)
```

    ('test', 0.9609375, 1.0, 0.09948365)
    ('test', 0.96875, 1.0, 0.06669818)
    ('test', 0.96875, 1.0, 0.067689285)
    ('test', 0.9765625, 1.0, 0.061369315)
    ('test', 0.953125, 1.0, 0.14121476)
    ('final test result', 0.971855, 0.9992822)


## 4. 量化

按照默认配置在``train_program``和``val_program``中加入量化和反量化op.


```python
quant_program = slim.quant.quant_aware(train_program, exe.place, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)
```

    2020-02-05 02:51:45,543-INFO: quant_aware config {'moving_rate': 0.9, 'weight_quantize_type': 'channel_wise_abs_max', 'is_full_quantize': False, 'dtype': 'int8', 'weight_bits': 8, 'window_size': 10000, 'activation_bits': 8, 'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'], 'not_quant_pattern': ['skip_quant'], 'activation_quantize_type': 'moving_average_abs_max', 'for_tensorrt': False}
    2020-02-05 02:51:46,960-INFO: quant_aware config {'moving_rate': 0.9, 'weight_quantize_type': 'channel_wise_abs_max', 'is_full_quantize': False, 'dtype': 'int8', 'weight_bits': 8, 'window_size': 10000, 'activation_bits': 8, 'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'], 'not_quant_pattern': ['skip_quant'], 'activation_quantize_type': 'moving_average_abs_max', 'for_tensorrt': False}


## 5. 训练和测试量化后的模型
训练


```python
train(quant_program)
```

    ('train', 0.9453125, 1.0, 0.15119126)
    ('train', 0.984375, 1.0, 0.04564587)
    ('train', 0.96875, 1.0, 0.10310037)
    ('train', 0.984375, 1.0, 0.070371866)
    ('train', 0.9609375, 1.0, 0.12202865)


测试


```python
test(val_quant_program)
```

    ('test', 0.984375, 1.0, 0.038413208)
    ('test', 0.9765625, 1.0, 0.045030694)
    ('test', 0.984375, 1.0, 0.043231864)
    ('test', 0.984375, 1.0, 0.05136764)
    ('test', 0.9765625, 1.0, 0.122583434)
    ('final test result', 0.98505944, 0.999783)

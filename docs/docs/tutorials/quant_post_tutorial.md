# 图像分类模型离线量化-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用PaddleSlim的[离线量化接口](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/docs/api/quantization_api.md)。 该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 训练模型
4. 离线量化

## 1. 导入依赖
PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:


```python
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 构建网络
该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。               为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：

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

def test(prog, outputs=outputs):
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

    ('train', 0.109375, 0.5625, 2.6159298)
    ('train', 0.953125, 1.0, 0.18042225)
    ('train', 0.9609375, 1.0, 0.15906152)
    ('train', 0.96875, 0.9921875, 0.1544011)
    ('train', 0.9609375, 1.0, 0.12998523)


测试分类网络


```python
test(val_program)
```

    ('test', 0.9453125, 1.0, 0.15452775)
    ('test', 0.9609375, 1.0, 0.081429556)
    ('test', 0.9921875, 1.0, 0.05349218)
    ('test', 0.9609375, 1.0, 0.14042228)
    ('test', 0.9375, 1.0, 0.20278816)
    ('final test result', 0.9669638, 0.998965)


保存inference model


```python
target_vars = [val_program.global_block().var(name) for name in outputs]
fluid.io.save_inference_model(dirname='./inference_model',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=val_program)
```




    [u'save_infer_model/scale_0',
     u'save_infer_model/scale_1',
     u'save_infer_model/scale_2']



## 4. 离线量化

使用10个batch的数据进行参数校正


```python
slim.quant.quant_post(
        executor=exe,
        model_dir='./inference_model',
        quantize_model_path='./quant_post_model',
        sample_generator=reader.test(),
        batch_nums=10)
```

    2020-02-05 03:33:41,164-INFO: run batch: 0
    2020-02-05 03:33:41,164-INFO: run batch: 0
    2020-02-05 03:33:41,452-INFO: run batch: 5
    2020-02-05 03:33:41,452-INFO: run batch: 5
    2020-02-05 03:33:41,582-INFO: all run batch: 10
    2020-02-05 03:33:41,582-INFO: all run batch: 10
    2020-02-05 03:33:41,585-INFO: calculate scale factor ...
    2020-02-05 03:33:41,585-INFO: calculate scale factor ...
    2020-02-05 03:33:53,840-INFO: update the program ...
    2020-02-05 03:33:53,840-INFO: update the program ...


加载量化后的模型进行测试


```python
quant_post_prog, feed_target_names, fetch_targets = fluid.io.load_inference_model(
        dirname='./quant_post_model',
        executor=exe)
test(quant_post_prog, fetch_targets)
```

    ('test', 0.9375, 1.0, 0.18027544)
    ('test', 0.984375, 1.0, 0.085289225)
    ('test', 0.984375, 1.0, 0.05745204)
    ('test', 0.9609375, 0.9921875, 0.13197306)
    ('test', 0.9375, 1.0, 0.22449243)
    ('final test result', 0.96285725, 0.9987814)

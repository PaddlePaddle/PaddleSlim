# 图像分类模型量化训练-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用PaddleSlim的[量化训练接口](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/docs/api/quantization_api.md)。 该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 训练模型
4. 量化
5. 训练和测试量化后的模型
6. 保存量化后的模型

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
该章节介绍了如何定义输入数据和如何训练和测试分类模型。先训练分类模型的原因是量化训练过程是在训练好的模型上进行的，也就是说是在训练好的模型的基础上加入量化反量化op之后，用小学习率进行参数微调。

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
先定义训练和测试函数，正常训练和量化训练时只需要调用函数即可。在训练函数中执行了一个epoch的训练，因为MNIST数据集数据较少，一个epoch就可将top1精度训练到95%以上。


```python
def train(prog):
    iter = 0
    for data in train_reader():
        acc1, acc5, loss = exe.run(prog, feed=train_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1
        
def test(prog):
    iter = 0
    res = [[], []]
    for data in train_reader():
        acc1, acc5, loss = exe.run(prog, feed=train_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('test iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result top1={}, top5={}'.format(np.array(res[0]).mean(), np.array(res[1]).mean()))
```

调用``train``函数训练分类网络，``train_program``是在第2步：构建网络中定义的。


```python
train(train_program)
```


调用``test``函数测试分类网络，``val_program``是在第2步：构建网络中定义的。


```python
test(val_program)
```


## 4. 量化

按照[默认配置](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#_1)在``train_program``和``val_program``中加入量化和反量化op.


```python
quant_program = slim.quant.quant_aware(train_program, exe.place, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)
```


## 5. 训练和测试量化后的模型
微调量化后的模型，训练一个epoch后测试。


```python
train(quant_program)
```


测试量化后的模型，和``3.2 训练和测试``中得到的测试结果相比，精度相近，达到了无损量化。


```python
test(val_quant_program)
```


## 6. 保存量化后的模型

在``4. 量化``中使用接口``slim.quant.quant_aware``接口得到的模型只适合训练时使用，为了得到最终使用时的模型，需要使用[slim.quant.convert](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#convert)接口，然后使用[fluid.io.save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/save_inference_model_cn.html#save-inference-model)保存模型。``float_prog``的参数数据类型是float32，但是数据范围是int8, 保存之后可使用fluid或者paddle-lite加载使用，paddle-lite在使用时，会先将类型转换为int8。``int8_prog``的参数数据类型是int8, 保存后可看到量化后模型大小，不可加载使用。


```python
float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)
target_vars = [float_prog.global_block().var(name) for name in outputs]
fluid.io.save_inference_model(dirname='./inference_model/float',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=float_prog)
fluid.io.save_inference_model(dirname='./inference_model/int8',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=int8_prog)
```

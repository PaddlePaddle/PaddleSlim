 # 图像分类模型离线量化-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用PaddleSlim的[离线量化接口](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post)。 该示例包含以下步骤：

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
该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：

>注意：paddleslim.models下的API并非PaddleSlim常规API，是为了简化示例而封装预定义的一系列方法，比如：模型结构的定义、Program的构建等。


```python
exe, train_program, val_program, inputs, outputs = \
    slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=True)
```

## 3. 训练模型
该章节介绍了如何定义输入数据和如何训练和测试分类模型。先训练分类模型的原因是离线量化需要一个训练好的模型。

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
先定义训练和测试函数。在训练函数中执行了一个epoch的训练，因为MNIST数据集数据较少，一个epoch就可将top1精度训练到95%以上。



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

调用``train``函数训练分类网络，``train_program``是在第2步：构建网络中定义的。


```python
train(train_program)
```


调用``test``函数测试分类网络，``val_program``是在第2步：构建网络中定义的。


```python
test(val_program)
```


保存inference model，将训练好的分类模型保存在``'./inference_model'``下，后续进行离线量化时将加载保存在此处的模型。


```python
target_vars = [val_program.global_block().var(name) for name in outputs]
fluid.io.save_inference_model(dirname='./inference_model',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=val_program)
```

## 4. 离线量化

调用离线量化接口，加载文件夹``'./inference_model'``训练好的分类模型，并使用10个batch的数据进行参数校正。此过程无需训练，只需跑前向过程来计算量化所需参数。离线量化后的模型保存在文件夹``'./quant_post_model'``下。


```python
slim.quant.quant_post(
        executor=exe,
        model_dir='./inference_model',
        quantize_model_path='./quant_post_model',
        sample_generator=reader.test(),
        batch_nums=10)
```


加载保存在文件夹``'./quant_post_model'``下的量化后的模型进行测试，可看到精度和``3.2 训练和测试``中得到的测试精度相近，因此离线量化过程对于此分类模型几乎无损。


```python
quant_post_prog, feed_target_names, fetch_targets = fluid.io.load_inference_model(
        dirname='./quant_post_model',
        executor=exe)
test(quant_post_prog, fetch_targets)
```

# 量化训练

该教程以图像分类模型MobileNetV1为例，说明如何快速使用PaddleSlim的[量化训练接口](../api_cn/quantization_api.html)。 该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 训练模型
4. 量化
5. 训练和测试量化后的模型
6. 保存量化后的模型

## 1. 导入依赖
PaddleSlim依赖Paddle2.0版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:


```python
import paddle
import paddleslim as slim
import numpy as np
paddle.enable_static()
```

## 2. 构建网络
该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。               为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：



```python
USE_GPU = True
model = slim.models.MobileNet()
train_program = paddle.static.Program()
startup = paddle.static.Program()
with paddle.static.program_guard(train_program, startup):
    image = paddle.static.data(
        name='image', shape=[None, 1, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    gt = paddle.reshape(label, [-1, 1])
    out = model.net(input=image, class_dim=10)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=gt)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=gt, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=gt, k=5)
    opt = paddle.optimizer.Momentum(0.01, 0.9)
    opt.minimize(avg_cost)

place = paddle.CUDAPlace(0) if USE_GPU else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(startup)
val_program = train_program.clone(for_test=True)
```

## 3. 训练模型
该章节介绍了如何定义输入数据和如何训练和测试分类模型。先训练分类模型的原因是量化训练过程是在训练好的模型上进行的，也就是说是在训练好的模型的基础上加入量化反量化op之后，用小学习率进行参数微调。

### 3.1 定义输入数据

为了快速执行该示例，我们选取简单的MNIST数据，Paddle框架的`paddle.vision.dataset`包定义了MNIST数据的下载和读取。
代码如下：


```python
import paddle.vision.transforms as T
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = paddle.vision.datasets.MNIST(
    mode="train", backend="cv2", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(
    mode="test", backend="cv2", transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=True)
test_loader = paddle.io.DataLoader(
    test_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=False)
```

### 3.2 训练和测试
先定义训练和测试函数，正常训练和量化训练时只需要调用函数即可。在训练函数中执行了一个epoch的训练，因为MNIST数据集数据较少，一个epoch就可将top1精度训练到95%以上。


```python
outputs = [acc_top1.name, acc_top5.name, avg_cost.name]
def train(prog):
    iter = 0
    for data in train_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1

def test(prog):
    iter = 0
    res = [[], []]
    for data in test_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
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

按照[默认配置](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#id2)在``train_program``和``val_program``中加入量化和反量化op.


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

在``4. 量化``中使用接口``slim.quant.quant_aware``接口得到的模型只适合训练时使用，为了得到最终使用时的模型，需要使用[slim.quant.convert](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#convert)接口，然后使用[fluid.io.save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/save_inference_model_cn.html#save-inference-model)保存模型。``float_prog``的参数数据类型是float32，但是数据范围是int8, 保存之后可使用fluid或者paddle-lite加载使用，paddle-lite在使用时，会先将类型转换为int8。``int8_prog``的参数数据类型是int8, 保存后可看到量化后模型大小，不可加载使用。


```python
float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)
target_vars = [float_prog.global_block().var(outputs[-1])]
paddle.static.save_inference_model(
        path_prefix='./inference_model/float',
        feed_vars=[image],
        fetch_vars=target_vars,
        executor=exe,
        program=float_prog)
paddle.static.save_inference_model(
        path_prefix='./inference_model/int8',
        feed_vars=[image],
        fetch_vars=target_vars,
        executor=exe,
        program=int8_prog)
```

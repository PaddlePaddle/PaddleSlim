#  卷积Filter剪裁

该教程以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的卷积通道剪裁接口](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html)。
该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 剪裁
4. 训练剪裁后的模型

以下章节依次次介绍每个步骤的内容。

## 1. 导入依赖

PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
paddle.enable_static()
```

## 2. 构建网络

该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。
为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：

```
exe, train_program, val_program, inputs, outputs =
    slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=False)
```

>注意：paddleslim.models下的API并非PaddleSlim常规API，是为了简化示例而封装预定义的一系列方法，比如：模型结构的定义、Program的构建等。

## 3. 剪裁卷积层通道

### 3.1 计算剪裁之前的FLOPs

```
FLOPs = slim.analysis.flops(train_program)
print("FLOPs: {}".format(FLOPs))
```

### 3.2 剪裁

我们这里对参数名为`conv2_1_sep_weights`和`conv2_2_sep_weights`的卷积层进行剪裁，分别剪掉20%和30%的通道数。
代码如下所示：

```
pruner = slim.prune.Pruner()
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2_1_sep_weights", "conv2_2_sep_weights"],
        ratios=[0.33] * 2,
        place=fluid.CPUPlace())
```

以上操作会修改`train_program`中对应卷积层参数的定义，同时对`fluid.global_scope()`中存储的参数数组进行裁剪。

### 3.3 计算剪裁之后的FLOPs

```
FLOPs = slim.analysis.flops(pruned_program)
print("FLOPs: {}".format(FLOPs))
```

## 4. 训练剪裁后的模型

### 4.1 定义输入数据

为了快速执行该示例，我们选取简单的MNIST数据，Paddle框架的`paddle.dataset.mnist`包定义了MNIST数据的下载和读取。
代码如下：

```
import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
```

### 4.2 执行训练
以下代码执行了一个`epoch`的训练：

```
for data in train_reader():
    acc1, acc5, loss, _ = exe.run(pruned_program, feed=train_feeder.feed(data), fetch_list=outputs)
    print(acc1, acc5, loss)
```

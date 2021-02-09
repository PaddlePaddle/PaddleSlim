#  敏感度分析

该教程以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的敏感度分析接口](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html#sensitivity)。
该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 定义输入数据
4. 定义模型评估方法
5. 训练模型
6. 获取待分析卷积参数名称
7. 分析敏感度
8. 剪裁模型

以下章节依次介绍每个步骤的内容。

## 1. 导入依赖

PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:


```python
import paddle
import paddle.fluid as fluid
import paddleslim as slim
paddle.enable_static()
```

## 2. 构建网络

该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。
为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：


```python
exe, train_program, val_program, inputs, outputs = slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=True)
place = fluid.CUDAPlace(0)
```

## 3 定义输入数据

为了快速执行该示例，我们选取简单的MNIST数据，Paddle框架的`paddle.dataset.mnist`包定义了MNIST数据的下载和读取。
代码如下：


```python
import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.fluid.io.batch(
        reader.test(), batch_size=128, drop_last=True)
data_feeder = fluid.DataFeeder(inputs, place)
```

## 4. 定义模型评估方法

在计算敏感度时，需要裁剪单个卷积层后的模型在测试数据上的效果，我们定义以下方法实现该功能：


```python
import numpy as np
def test(program):
    acc_top1_ns = []
    acc_top5_ns = []
    for data in test_reader():
        acc_top1_n, acc_top5_n, _, _ = exe.run(
            program,
            feed=data_feeder.feed(data),
            fetch_list=outputs)
        acc_top1_ns.append(np.mean(acc_top1_n))
        acc_top5_ns.append(np.mean(acc_top5_n))
    print("Final eva - acc_top1: {}; acc_top5: {}".format(
        np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
    return np.mean(np.array(acc_top1_ns))
```

## 5. 训练模型

只有训练好的模型才能做敏感度分析，因为该示例任务相对简单，我这里用训练一个`epoch`产出的模型做敏感度分析。对于其它训练比较耗时的模型，您可以加载训练好的模型权重。

以下为模型训练代码：


```python
for data in train_reader():
    acc1, acc5, loss = exe.run(train_program, feed=data_feeder.feed(data), fetch_list=outputs)
print(np.mean(acc1), np.mean(acc5), np.mean(loss))
```

用上节定义的模型评估方法，评估当前模型在测试集上的精度：


```python
test(val_program)
```

## 6. 获取待分析卷积参数

```python
params = []
for param in train_program.global_block().all_parameters():
    if "_sep_weights" in param.name:
        params.append(param.name)
print(params)
params = params[:5]
```

## 7. 分析敏感度

### 7.1 简单计算敏感度

调用[sensitivity接口](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html#paddleslim.prune.sensitivity)对训练好的模型进行敏感度分析。

在计算过程中，敏感度信息会不断追加保存到选项`sensitivities_file`指定的文件中，该文件中已有的敏感度信息不会被重复计算。

先用以下命令删除当前路径下可能已有的`sensitivities_0.data`文件:


```python
!rm -rf sensitivities_0.data
```

除了指定待分析的卷积层参数，我们还可以指定敏感度分析的粒度和范围，即单个卷积层参数分别被剪裁掉的比例。

如果待分析的模型比较敏感，剪掉单个卷积层的40%的通道，模型在测试集上的精度损失就达90%，那么`pruned_ratios`最大设置到0.4即可，比如：
`[0.1, 0.2, 0.3, 0.4]`

为了得到更精确的敏感度信息，我可以适当调小`pruned_ratios`的粒度，比如：`[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]`

`pruned_ratios`的粒度越小，计算敏感度的速度越慢。


```python
sens_0 = slim.prune.sensitivity(
        val_program,
        place,
        params,
        test,
        sensitivities_file="sensitivities_0.data",
        pruned_ratios=[0.1, 0.2])
print(sens_0)
```

### 7.2 扩展敏感度信息

第7.1节计算敏感度用的是`pruned_ratios=[0.1, 0.2]`, 我们可以在此基础上将其扩展到`[0.1, 0.2, 0.3]`


```python
sens_0 = slim.prune.sensitivity(
        val_program,
        place,
        params,
        test,
        sensitivities_file="sensitivities_0.data",
        pruned_ratios=[0.3])
print(sens_0)
```

### 7.3 多进程加速计算敏感度信息

敏感度分析所用时间取决于待分析的卷积层数量和模型评估的速度，我们可以通过多进程的方式加速敏感度计算。

在不同的进程设置不同`pruned_ratios`, 然后将结果合并。

#### 7.3.1 多进程计算敏感度

在以上章节，我们计算了`pruned_ratios=[0.1, 0.2, 0.3]`的敏感度，并将其保存到了文件`sensitivities_0.data`中。

在另一个进程中，我们可以设置`pruned_ratios=[0.4]`，并将结果保存在文件`sensitivities_1.data`中。代码如下：


```python
sens_1 = slim.prune.sensitivity(
        val_program,
        place,
        params,
        test,
        sensitivities_file="sensitivities_1.data",
        pruned_ratios=[0.4])
print(sens_1)
```

#### 7.3.2 加载多个进程产出的敏感度文件

```python
s_0 = slim.prune.load_sensitivities("sensitivities_0.data")
s_1 = slim.prune.load_sensitivities("sensitivities_1.data")
print(s_0)
print(s_1)
```

#### 7.3.3 合并敏感度信息


```python
s = slim.prune.merge_sensitive([s_0, s_1])
print(s)
```

## 8. 剪裁模型

根据以上章节产出的敏感度信息，对模型进行剪裁。

### 8.1 计算剪裁率

首先，调用PaddleSlim提供的[get_ratios_by_loss](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html#paddleslim.prune.get_ratios_by_loss)方法根据敏感度计算剪裁率，通过调整参数`loss`大小获得合适的一组剪裁率：


```python
loss = 0.01
ratios = slim.prune.get_ratios_by_loss(s_0, loss)
print(ratios)
```

### 8.2 剪裁训练网络


```python
pruner = slim.prune.Pruner()
print("FLOPs before pruning: {}".format(slim.analysis.flops(train_program)))
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=ratios.keys(),
        ratios=ratios.values(),
        place=place)
print("FLOPs after pruning: {}".format(slim.analysis.flops(pruned_program)))
```

### 8.3 剪裁测试网络

>注意：对测试网络进行剪裁时，需要将`only_graph`设置为True，具体原因请参考[Pruner API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html#paddleslim.prune.Pruner)


```python
pruner = slim.prune.Pruner()
print("FLOPs before pruning: {}".format(slim.analysis.flops(val_program)))
pruned_val_program, _, _ = pruner.prune(
        val_program,
        fluid.global_scope(),
        params=ratios.keys(),
        ratios=ratios.values(),
        place=place,
        only_graph=True)
print("FLOPs after pruning: {}".format(slim.analysis.flops(pruned_val_program)))
```

测试一下剪裁后的模型在测试集上的精度：

```python
test(pruned_val_program)
```

### 8.4 训练剪裁后的模型

对剪裁后的模型在训练集上训练一个`epoch`:


```python
for data in train_reader():
    acc1, acc5, loss, _ = exe.run(pruned_program, feed=data_feeder.feed(data), fetch_list=outputs)
print(np.mean(acc1), np.mean(acc5), np.mean(loss))
```

测试训练后模型的精度：

```python
test(pruned_val_program)
```

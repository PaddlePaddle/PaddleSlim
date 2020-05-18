# Pruning of image classification model - sensitivity

In this tutorial, you will learn how to use [sensitivity API of PaddleSlim](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#sensitivity) by a demo of MobileNetV1 model on MNIST dataset。
This tutorial following workflow:

1. Import dependency
2. Build model
3. Define data reader
4. Define function for test
5. Training model
6. Get names of parameter
7. Compute sensitivities
8. Pruning model


## 1. Import dependency

PaddleSlim dependents on Paddle1.7. Please ensure that you have installed paddle correctly. Import Paddle and PaddleSlim as below:

```python
import paddle
import paddle.fluid as fluid
import paddleslim as slim
```

## 2. Build model

This section will build a classsification model based `MobileNetV1` for MNIST task. The shape of the input is `[1, 28, 28]` and the output number is 10.

To make the code simple, we define a function in package `paddleslim.models` to build classification model.
Excute following code to build a model,


```python
exe, train_program, val_program, inputs, outputs = slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=True)
place = fluid.CUDAPlace(0)
```

>Note：The functions in paddleslim.models is just used in tutorials or demos.

## 3 Define data reader

MNIST dataset is used for making the demo can be executed quickly. It defines some functions for downloading and reading MNIST dataset in package `paddle.dataset.mnist`.
Show as below：

```python
import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.fluid.io.batch(
        reader.test(), batch_size=128, drop_last=True)
data_feeder = fluid.DataFeeder(inputs, place)
```

## 4. Define test function

To get the performance of model on test dataset after pruning a convolution layer, we define a test function as below:

```python
import numpy as np
def test(program):
    acc_top1_ns = []
    acc_top5_ns = []
    for data in test_reader():
        acc_top1_n, acc_top5_n, _ = exe.run(
            program,
            feed=data_feeder.feed(data),
            fetch_list=outputs)
        acc_top1_ns.append(np.mean(acc_top1_n))
        acc_top5_ns.append(np.mean(acc_top5_n))
    print("Final eva - acc_top1: {}; acc_top5: {}".format(
        np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
    return np.mean(np.array(acc_top1_ns))
```

## 5. Training model

Sensitivity analysis is dependent on pretrained model. So we should train the model defined in section 2 for some epochs. One epoch training is enough for this simple demo while more epochs may be necessary for other model. Or you can load pretrained model from filesystem.

Training model as below:


```python
for data in train_reader():
    acc1, acc5, loss = exe.run(train_program, feed=data_feeder.feed(data), fetch_list=outputs)
print(np.mean(acc1), np.mean(acc5), np.mean(loss))
```

Get the performance using the test function defined in section 4:

```python
test(val_program)
```

## 6. Get names of parameters

```python
params = []
for param in train_program.global_block().all_parameters():
    if "_sep_weights" in param.name:
        params.append(param.name)
print(params)
params = params[:5]
```

## 7. Compute sensitivities

### 7.1 Compute in single process

Apply sensitivity analysis on pretrained model by calling [sensitivity API](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#sensitivity).

The sensitivities will be appended into the file given by option `sensitivities_file` during computing.
The information in this file won`t be computed repeatedly.

Remove the file `sensitivities_0.data` in current directory:

```python
!rm -rf sensitivities_0.data
```

Apart from the parameters to be analyzed, it also support for setting the ratios that each convolutoin will be pruned.

If one model losses 90% accuracy on test dataset when its single convolution layer is pruned by 40%, then we can set `pruned_ratios` to `[0.1, 0.2, 0.3, 0.4]`.

The granularity of `pruned_ratios` should be small to get more reasonable sensitivities. But small granularity of `pruned_ratios` will slow down the computing.

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

### 7.2 Expand sensitivities

We can expand `pruned_ratios` to `[0.1, 0.2, 0.3]` based the sensitivities generated in section 7.1.

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

### 7.3 Computing sensitivity in multi-process

The time cost of computing sensitivities is dependent on the count of parameters and the speed of model evaluation on test dataset. We can speed up computing by multi-process.

Split `pruned_ratios` into multi-process, and merge the sensitivities from multi-process.

#### 7.3.1 Computing in each process

We have compute the sensitivities when `pruned_ratios=[0.1, 0.2, 0.3]` and saved the sensitivities into file named `sensitivities_0.data`.

在另一个进程中，The we start a task by setting `pruned_ratios=[0.4]` in another process and save result into file named `sensitivities_1.data`. Show as below：


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

#### 7.3.2 Load sensitivity file generated in multi-process

```python
s_0 = slim.prune.load_sensitivities("sensitivities_0.data")
s_1 = slim.prune.load_sensitivities("sensitivities_1.data")
print(s_0)
print(s_1)
```

#### 7.3.3 Merge sensitivies


```python
s = slim.prune.merge_sensitive([s_0, s_1])
print(s)
```

## 8. Pruning model

Pruning model according to the sensitivities generated in section 7.3.3.

### 8.1 Get pruning ratios

Get a group of ratios by calling [get_ratios_by_loss](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#get_ratios_by_loss) fuction：


```python
loss = 0.01
ratios = slim.prune.get_ratios_by_loss(s_0, loss)
print(ratios)
```

### 8.2 Pruning training network


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

### 8.3 Pruning test network

Note：The `only_graph` should be set to True while pruning test network. [Pruner API](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#pruner)


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

Get accuracy of pruned model on test dataset:

```python
test(pruned_val_program)
```

### 8.4 Training pruned model

Training pruned model:


```python
for data in train_reader():
    acc1, acc5, loss = exe.run(pruned_program, feed=data_feeder.feed(data), fetch_list=outputs)
print(np.mean(acc1), np.mean(acc5), np.mean(loss))
```

Get accuracy of model after training:

```python
test(pruned_val_program)
```

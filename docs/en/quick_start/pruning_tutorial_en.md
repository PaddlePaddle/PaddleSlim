# Channel Pruning for Image Classification

In this tutorial, you will learn how to use channel pruning API of PaddleSlim
by a demo of MobileNetV1 model on MNIST dataset. This tutorial following workflow:

1. Import dependency
2. Build model
3. Prune model
4. Train pruned model

## 1. Import dependency

PaddleSlim dependents on Paddle1.7. Please ensure that you have installed paddle correctly. Import Paddle and PaddleSlim as below:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
```

## 2. Build Model

This section will build a classsification model based `MobileNetV1` for MNIST task. The shape of the input is `[1, 28, 28]` and the output number is 10.

To make the code simple, we define a function in package `paddleslim.models` to build classification model.
Excute following code to build a model,

```
exe, train_program, val_program, inputs, outputs =
    slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=False)
```

>Note：The functions in paddleslim.models is just used in tutorials or demos.

## 3. Prune model

### 3.1 Compute FLOPs bofore pruning

```
FLOPs = slim.analysis.flops(train_program)
print("FLOPs: {}".format(FLOPs))
```

### 3.2 Pruning

The section will prune the parameters named `conv2_1_sep_weights` and `conv2_2_sep_weights` by 20% and 30%.

```
pruner = slim.prune.Pruner()
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2_1_sep_weights", "conv2_2_sep_weights"],
        ratios=[0.33] * 2,
        place=fluid.CPUPlace())
```

It will change the shapes of parameters defined in `train_program`. And the parameters` values stored in `fluid.global_scope()` will be pruned.


### 3.3 Compute FLOPs after pruning

```
FLOPs = paddleslim.analysis.flops(train_program)
print("FLOPs: {}".format(FLOPs))
```

## 4. Train pruned model

### 4.1 Define dataset

To make you easily run this demo, it will training on MNIST dataset. The package `paddle.dataset.mnist` of Paddle defines the downloading and reading of MNIST dataset.
Define training data reader and test data reader as below：

```
import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
```

### 4.2 Training

Excute following code to run an `epoch` training:

```
for data in train_reader():
    acc1, acc5, loss = exe.run(pruned_program, feed=train_feeder.feed(data), fetch_list=outputs)
    print(acc1, acc5, loss)
```

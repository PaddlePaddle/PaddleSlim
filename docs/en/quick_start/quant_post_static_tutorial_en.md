# Post-training Quantization of image classification model - quick start

This tutorial shows how to do post training quantization using [API](https://paddlepaddle.github.io/PaddleSlim/api_en/paddleslim.quant.html#paddleslim.quant.quanter.quant_post) in PaddleSlim. We use MobileNetV1 to train image classification model as example. The tutorial contains follow sections:

1. Necessary imports
2. Model architecture
3. Train normal model
4. Post training quantization

## 1. Necessary imports
PaddleSlim depends on Paddle1.7. Please make true that you have installed Paddle correctly. Then do the necessary imports:


```python
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```
## 2. Model architecture

The section constructs a classification model, which use ``MobileNetV1`` and MNIST dataset. The model's input size is `[1, 28, 28]` and output size is 10. In order to show tutorial conveniently, we pre-defined a method to get image classification model in `paddleslim.models`.

>note: The APIs in `paddleslim.models` are not formal inferface in PaddleSlim. They are defined to simplify the tutorial such as the definition of model structure and the construction of Program.


```python
exe, train_program, val_program, inputs, outputs = \
    slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=True)
```

## 3. Train normal model

The section shows how to define model inputs, train and test model. The reason for training the normal image classification model first is that the post training quantization is performed on the well-trained model.

### 3.1 input data definition

To speed up training process, we select MNIST dataset to train image classification model. The API `paddle.dataset.mnist` in Paddle framework contains downloading and reading the images in dataset.

```python
import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.fluid.io.batch(
cs/en/quick_start/quant_aware_tutorial_en.md
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
```

### 3.2 training model and testing

Define functions to train and test model. We only need call the functions when formal model training and quantization model training. The function does one epoch training because that MNIST dataset is small and top1 accuracy will reach 95% after one epoch.

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

Call ``train`` function to train normal classification model. ``train_program`` is defined in 2. Model architecture.


```python
train(train_program)
```

Call ``test`` function to test normal classification model. ``val_program`` is defined in 2. Model architecture.

```python
test(val_program)
```


Save inference model. Save well-trained model in ``'./inference_model'``. We will load the model when doing post training quantization.


```python
target_vars = [val_program.global_block().var(name) for name in outputs]
fluid.io.save_inference_model(dirname='./inference_model',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=val_program)
```

## 4. Post training quantization

Call ``slim.quant.quant_post`` API to do post training quantization. The API will load the inference model in ``'./inference_model'`` first and calibrate the quantization parameters using data in sample_generator. In this tutorial, we use 10 mini-batch data to calibrate the quantization parameters. There is no need to train model but run forward to get activations for quantization scales calculation. The model after post training quantization are saved in ``'./quant_post_model'``.

```python
slim.quant.quant_post(
        executor=exe,
        model_dir='./inference_model',
        quantize_model_path='./quant_post_model',
        sample_generator=reader.test(),
        batch_nums=10)
```

Load the model after post training quantization in ``'./quant_post_model'`` and run ``test`` function. The top1 and top5 accuracy are close to result in ``3.2 training model and testing``. We preform the post training quantization without loss on this image classification model.

```python
quant_post_prog, feed_target_names, fetch_targets = fluid.io.load_inference_model(
        dirname='./quant_post_model',
        model_filename='__model__',
        params_filename='__params__',
        executor=exe)
test(quant_post_prog, fetch_targets)
```

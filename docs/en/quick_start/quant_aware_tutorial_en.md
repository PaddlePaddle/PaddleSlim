# Training-aware Quantization of image classification model - quick start

This tutorial shows how to do training-aware quantization using [API](https://paddlepaddle.github.io/PaddleSlim/api_en/paddleslim.quant.html#paddleslim.quant.quanter.quant_aware) in PaddleSlim. We use MobileNetV1 to train image classification model as example. The tutorial contains follow sections:

1. Necessary imports
2. Model architecture
3. Train normal model
4. Quantization
5. Train model after quantization
6. Save model after quantization

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

The section shows how to define model inputs, train and test model. The reason for training the normal image classification model first is that the quantization model's training process is performed on the well-trained model. We add quantization and dequantization operators in well-trained model and finetune using smaller learning rate.

### 3.1 input data definition

To speed up training process, we select MNIST dataset to train image classification model. The API `paddle.dataset.mnist` in Paddle framework contains downloading and reading the images in dataset.

```python
import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.fluid.io.batch(
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

Call ``train`` function to train normal classification model. ``train_program`` is defined in 2. Model architecture.

```python
train(train_program)
```

Call ``test`` function to test normal classification model. ``val_program`` is defined in 2. Model architecture.

```python
test(val_program)
```


## 4. Quantization

We call ``quant_aware`` API to add quantization and dequantization operators in ``train_program`` and ``val_program`` according to [default configuration](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#id2).

```python
quant_program = slim.quant.quant_aware(train_program, exe.place, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)
```


## 5. Train model after quantization

Finetune the model after quantization. Test model after one epoch training.

```python
train(quant_program)
```

Test model after quantization. The top1 and top5 accuracy are close to result in ``3.2 training model and testing``. We preform the training-aware quantization without loss on this image classification model.

```python
test(val_quant_program)
```


## 6. Save model after quantization

The model in ``4. Quantization`` after calling ``slim.quant.quant_aware`` API is only suitable to train. To get the inference model, we should use [slim.quant.convert](https://paddlepaddle.github.io/PaddleSlim/api_en/paddleslim.quant.html#paddleslim.quant.quanter.convert) API to change model architecture and use [fluid.io.save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/save_inference_model_cn.html#save-inference-model) to save model. ``float_prog``'s parameters are float32 dtype but in int8's range which can be used in ``fluid`` or ``paddle-lite``. ``paddle-lite`` will change the parameters' dtype from float32 to int8 first when loading the inference model. ``int8_prog``'s parameters are int8 dtype and we can get model size after quantization by saving it. ``int8_prog`` cannot be used in ``fluid`` or ``paddle-lite``.


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

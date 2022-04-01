# 低比特离线量化

## 动态模式

动态离线量化，将模型中特定OP的权重从FP32类型量化成INT8/16类型。

量化前需要有训练好的预测模型，可以根据需要将模型转化为INT8或INT16类型，目前只支持反量化预测方式，主要可以减小模型大小，对特定加载权重费时的模型可以起到一定加速效果。

- 权重量化成INT16类型，模型精度不受影响，模型大小为原始的1/2。
- 权重量化成INT8类型，模型精度会受到影响，模型大小为原始的1/4。

### 使用方法

- 准备预测模型：先保存好FP32的预测模型，用于量化压缩。
- 产出量化模型：使用PaddlePaddle调用动态离线量化离线量化接口，产出量化模型。

主要代码实现如下：

```python
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as reader
from paddleslim.models import MobileNet
from paddleslim.quant import quant_post_dynamic

paddle.enable_static()
image = paddle.static.data(name='image', shape=[None, 1, 28, 28], dtype='float32')
model = MobileNet()
out = model.net(input=image, class_dim=10)
main_prog = paddle.static.default_main_program()
val_prog = main_prog.clone(for_test=True)
place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())

paddle.fluid.io.save_inference_model(
    dirname='./model_path',
    feeded_var_names=[image.name],
    target_vars=[out],
    main_program=val_prog,
    executor=exe,
    model_filename='__model__',
    params_filename='__params__')

quant_post_dynamic(
        model_dir='./model_path',
        save_model_dir='./save_path',
        model_filename='__model__',
        params_filename='__params__',
        save_model_filename='__model__',
        save_params_filename='__params__')
```

## 静态离线量化

静态离线量化是基于采样数据，采用KL散度等方法计算量化比例因子的方法。相比量化训练，静态离线量化不需要重新训练，可以快速得到量化模型。

静态离线量化的目标是求取量化比例因子，主要有两种方法：非饱和量化方法 ( No Saturation) 和饱和量化方法 (Saturation)。非饱和量化方法计算FP32类型Tensor中绝对值的最大值`abs_max`，将其映射为127，则量化比例因子等于`abs_max/127`。饱和量化方法使用KL散度计算一个合适的阈值`T` (`0<T<mab_max`)，将其映射为127，则量化比例因子等于`T/127`。一般而言，对于待量化op的权重Tensor，采用非饱和量化方法，对于待量化op的激活Tensor（包括输入和输出），采用饱和量化方法 。

### 使用方法

静态离线量化的实现步骤如下：

- 加载预训练的FP32模型，配置reader；
- 读取样本数据，执行模型的前向推理，保存待量化op激活Tensor的数值；
- 基于激活Tensor的采样数据，使用饱和量化方法计算它的量化比例因子；
- 模型权重Tensor数据一直保持不变，使用非饱和方法计算它每个通道的绝对值最大值，作为每个通道的量化比例因子；
- 将FP32模型转成INT8模型，进行保存。

主要代码实现如下：

```python
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as reader
from paddleslim.models import MobileNet
from paddleslim.quant import quant_post_static

paddle.enable_static()
val_reader = reader.test()
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
paddle.enable_static()
image = paddle.static.data(name='image', shape=[None, 1, 28, 28], dtype='float32')
model = MobileNet()
out = model.net(input=image, class_dim=10)
main_prog = paddle.static.default_main_program()
val_prog = main_prog.clone(for_test=True)
place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())

paddle.fluid.io.save_inference_model(
    dirname='./model_path',
    feeded_var_names=[image.name],
    target_vars=[out],
    main_program=val_prog,
    executor=exe,
    model_filename='__model__',
    params_filename='__params__')
quant_post_static(
        executor=exe,
        model_dir='./model_path',
        quantize_model_path='./save_path',
        sample_generator=val_reader,
        model_filename='__model__',
        params_filename='__params__',
        batch_size=16,
        batch_nums=10)
```

详细代码与例程请参考：[静态离线量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)

### 实验结果

|       模型        |     压缩方法     | 原模型Top-1/Top-5 Acc | 量化模型Top-1/Top-5 Acc |
| :---------------: | :--------------: | :-------------------: | :---------------------: |
|    MobileNetV1    |   quant_post_static    |     70.99%/89.65%     |      70.18%/89.25%      |
|    MobileNetV2    |   quant_post_static    |     72.15%/90.65%     |      71.15%/90.11%      |
|     ResNet50      |   quant_post_static    |     76.50%/93.00%     |      76.33%/93.02%      |

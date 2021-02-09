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
import paddleslim
model_dir = path/to/fp32_model_params
save_model_dir = path/to/save_model_path
paddleslim.quant.quant_post_dynamic(model_dir=model_dir,
                   save_model_dir=save_model_dir,
                   weight_bits=8,
                   quantizable_op_type=['conv2d', 'mul'],
                   weight_quantize_type="channel_wise_abs_max",
                   generate_test_model=False)
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
import paddleslim
exe = paddle.static.Executor(place)
paddleslim.quant.quant_post(
  executor=exe,
  model_dir=model_path,
  quantize_model_path=save_path,
  sample_generator=reader,
  model_filename=model_filename,
  params_filename=params_filename,
  batch_nums=batch_num)
```

详细代码与例程请参考：[静态离线量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)

### 实验结果

|       模型        |     压缩方法     | 原模型Top-1/Top-5 Acc | 量化模型Top-1/Top-5 Acc |
| :---------------: | :--------------: | :-------------------: | :---------------------: |
|    MobileNetV1    |   quant_post_static    |     70.99%/89.65%     |      70.18%/89.25%      |
|    MobileNetV2    |   quant_post_static    |     72.15%/90.65%     |      71.15%/90.11%      |
|     ResNet50      |   quant_post_static    |     76.50%/93.00%     |      76.33%/93.02%      |

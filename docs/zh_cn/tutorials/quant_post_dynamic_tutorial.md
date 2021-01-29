# 动态离线量化

动态离线量化，将模型中特定OP的权重从FP32类型量化成INT8/16类型。

量化前需要有训练好的预测模型，可以根据需要将模型转化为INT8或INT16类型，目前只支持反量化预测方式，主要可以减小模型大小，对特定加载权重费时的模型可以起到一定加速效果

- 权重量化成INT16类型，模型精度不受影响，模型大小为原始的1/2
- 权重量化成INT8类型，模型精度会受到影响，模型大小为原始的1/4

## 使用方法

- 准备预测模型：先保存好FP32的预测模型，用于量化压缩
- 产出量化模型：使用PaddlePaddle调用动态离线量化离线量化接口，产出量化模型

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

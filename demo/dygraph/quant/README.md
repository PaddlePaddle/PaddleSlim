# 动态图量化训练

本示例介绍如何对动态图模型进行量化训练，示例以常用的MobileNetV1和MobileNetV3模型为例，介绍如何对其进行量化训练。


## 分类模型的量化训练流程

### 准备数据

在当前目录下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型

- 对于paddle vision支持的[模型](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/vision/models)：`[lenet, mobilenetv1, mobilenetv2, resnet, vgg]`可以直接使用vision内置的模型定义和ImageNet预训练权重
- 对于paddle vision暂未支持的模型，例如mobilenetv3，需要自行定义好模型结构以及准备相应的预训练权重
  - 本示例使用的是经过蒸馏的mobilenetv3模型，在ImageNet数据集上Top1精度达到78.96: [预训练权重下载](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar)


### 配置量化参数

```python
quant_config = {
    'weight_preprocess_type': None,
    'activation_preprocess_type': None,
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}
```

- `'weight_preprocess_type'`：代表对量化模型权重参数预处理的方法，目前支持PACT方法，如需使用可以改为'PACT'；默认为None，代表不对权重进行任何预处理。

- `'activation_preprocess_type'`：代表对量化模型激活值预处理的方法，目前支持PACT方法，如需使用可以改为'PACT'；默认为None，代表不对激活值进行任何预处理。

- `weight_quantize_type`：代表模型权重的量化方式，可选的有['abs_max', 'moving_average_abs_max', 'channel_wise_abs_max']，默认为channel_wise_abs_max

- `activation_quantize_type`：代表模型激活值的量化方式，可选的有['abs_max', 'moving_average_abs_max']，默认为moving_average_abs_max

- `quantizable_layer_type`：代表量化OP的类型，目前支持Conv2D和Linear



### 插入量化算子，得到量化训练模型

```python
quanter = QAT(config=quant_config)
quanter.quantize(net)
```

### 量化训练结束，保存量化模型

```python
quanter.save_quantized_model(net, 'save_dir', input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])
```

### 训练命令

- MobileNetV1

  我们使用普通的量化训练方法即可，启动命令如下：

   ```bash
  # 单卡训练
  python train.py --model=mobilenet_v1
  # 多卡训练，以0到3号卡为例
  python -m paddle.distributed.launch --gpus="0,1,2,3" train.py --model=mobilenet_v1
   ```
- MobileNetV3

  对于MobileNetV3，直接使用普通的量化损失较大，为降低量化损失，可以使用PACT的量化方法，启动命令如下：

  ```bash
  # 单卡训练
  python train.py  --lr=0.001 --use_pact=True --num_epochs=30 --l2_decay=2e-5 --ls_epsilon=0.1
  # 多卡训练，以0到3号卡为例
  python -m paddle.distributed.launch --gpus="0,1,2,3" train.py  --lr=0.001 --use_pact=True --num_epochs=30 --l2_decay=2e-5 --ls_epsilon=0.1
  ```



### 量化结果

| 模型        | FP32模型准确率（Top1/Top5） | 量化方法     | 量化模型准确率（Top1/Top5） |
| ----------- | --------------------------- | ------------ | --------------------------- |
| MobileNetV1 | 70.99/89.65                 | 普通在线量化 | 70.63/89.65                 |
| MobileNetV3 | 78.96/94.48                 | PACT在线量化 | 77.52/93.77                 |

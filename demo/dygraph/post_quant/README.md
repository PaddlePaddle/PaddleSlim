# 动态图离线量化

本示例介绍如何对动态图模型进行离线量化，示例以常用的MobileNetV1和MobileNetV3模型为例，介绍如何对其进行离线量化。


## 分类模型的离线量化流程

#### 准备数据

在当前目录下创建``data``文件夹，将``ImageNet``的验证集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

#### 准备需要离线量化的模型

- 对于paddle vision支持的[模型](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/vision/models)：`[lenet, mobilenetv1, mobilenetv2, resnet, vgg]`可以直接使用vision内置的模型定义和ImageNet预训练权重
- 对于paddle vision暂未支持的模型，例如mobilenetv3，需要自行定义好模型结构以及准备相应的预训练权重
  - 本示例使用的是的mobilenetv3模型，在ImageNet数据集上Top1精度达到75.0: [预训练权重下载](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams)


#### 初始化离线量化接口

- 保持默认离线量化配置
```python
ptq = PTQ()
```

- 如mobilenetv3等模型，需修改默认离线量化配置，按照如下方式修改：
```python
ptq_config = {'activation_quantizer': 'HistQuantizer', 'upsample_bins': 127, 'hist_percent': 0.999}
ptq = PTQ(**ptq_config)
```


#### 得到离线量化模型

通过ptq.quantize接口即可得到离线量化后的模型结构

```python
quant_model = ptq.quantize(fp32_model)
```

如果需要对模型进行fuse融合，设置`fuse=True`即可，fuse_list默认是None，表示对网络所有层进行fuse优化，如果需要自定义fuse某些层，可根据如下方式增加`fuse_list`，fuse后的模型更小，推理可能更快，精度持平或可能降低。
```python
count = 0
fuse_list = []
for name, layer in fp32_model.named_sublayers():
    if isinstance(layer, nn.Conv2D):
        fuse_list.append([name])
    if isinstance(layer, nn.BatchNorm2D):
        fuse_list[count].append(name)
    count += 1
quant_model = ptq.quantize(fp32_model, fuse=True, fuse_list=fuse_list)
```

#### 校准模型

```python
calibrate(quant_model, val_dataset, FLAGS.quant_batch_num,
              FLAGS.quant_batch_size)
```

## 启动命令

- MobileNetV1

   ```bash
  python3.7 ptq.py \
        --data=dataset/ILSVRC2012/ \
        --model=mobilenet_v1 \
        --quant_batch_num=10 \
        --quant_batch_size=10 \
        --output_dir="output_ptq"
   ```
- MobileNetV3

  对于MobileNetV3，直接使用默认离线量化配置进行校准，精度损失较大，为降低量化损失，在代码中默认设置了`skip_se_quant=True`，将`SE模块`跳过量化，并且调整batch_size和激活量化方式，启动命令如下：

  ```bash
  python3.7 ptq.py \
        --data=dataset/ILSVRC2012/ \
        --model=mobilenet_v3 \
        --pretrain_weight=MobileNetV3_large_x1_0_pretrained.pdparams \
        --quant_batch_num=10 \
        --quant_batch_size=32 \
        --output_dir="output_ptq"
  ```

## 评估精度

```bash
python3.7 eval.py --model_path=output_ptq/mobilenet_v3/int8_infer/ --data_dir=dataset/ILSVRC2012/ --use_gpu=True
```

- 评估时支持CPU，并且不依赖TensorRT，MKLDNN。


## 量化结果

| 模型        | FP32模型准确率（Top1/Top5） | 量化方法     | 量化模型准确率（Top1/Top5） |
| ----------- | --------------------------- | ------------ | --------------------------- |
| MobileNetV1 | 70.82/89.63                 | 离线量化 | 70.49/89.41                 |
| MobileNetV3 | 74.98/92.13                 | 离线量化 | 71.14/90.17               |

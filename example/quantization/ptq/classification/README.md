# 动态图离线量化

本示例介绍如何对动态图模型进行离线量化，示例以常用的MobileNetV1和MobileNetV3模型为例，介绍如何对其进行离线量化。


## 分类模型的离线量化流程

#### 准备数据

在当前目录下创建``data``文件夹，将``ImageNet``的验证集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

#### 准备需要离线量化的模型

本示例直接使用[paddle vision](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/vision/models)内置的模型结构和预训练权重。通过以下命令查看支持的所有模型：

```
python ptq.py --help
```

## 启动命令
以MobileNetV1为例，通过以下脚本启动PTQ任务：

```bash
python ptq.py \
      --data=dataset/ILSVRC2012/ \
      --model=mobilenet_v1 \
      --activation_observer='mse' \
      --weight_observer='mse_channel_wise' \
      --quant_batch_num=10 \
      --quant_batch_size=10 \
      --output_dir="output_ptq"
```

其中，通过 `activation_observer` 配置用于激活的量化算法，通过 `weight_observer` 配置用于权重的量化算法。
更多支持的量化算法，请执行以下命令查看：

```
python ptq.py --help
```

## 评估精度

执行以下命令，使用 PaddleInference 推理库测试推理精度：

```bash
python eval.py --model_path=output_ptq/mobilenet_v1/int8_infer/ --data_dir=dataset/ILSVRC2012/ --use_gpu=True
```

- 评估时支持CPU，并且不依赖TensorRT，MKLDNN。


## 量化结果

| 模型        | FP32模型准确率（Top1/Top5） | 量化方法（activation/weight）     | 量化模型准确率（Top1/Top5） |
| ----------- | --------------------------- | ------------ | --------------------------- |
| MobileNetV1 | 70.10%/90.10%                 | mse / mes_channel_wise | 69.10%/89.80%                |
| MobileNetV2 | 71.10%/90.90%                 | mse / mes_channel_wise | 70.70%/90.10%               |

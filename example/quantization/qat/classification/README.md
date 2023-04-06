# 动态图量化训练

本示例介绍如何对动态图模型进行量化训练，示例以常用的MobileNetV1，介绍如何对其进行量化训练。


## 分类模型的量化训练流程

### 准备数据

在当前目录下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型

本示例直接使用[paddle vision](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/vision/models)内置的模型结构和预训练权重。通过以下命令查看支持的所有模型：

```
python train.py --help
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

### 量化结果

| 模型        | FP32模型准确率（Top1/Top5） | 量化方法     | 量化模型准确率（Top1/Top5） |
| ----------- | --------------------------- | ------------ | --------------------------- |
| MobileNetV1 | 70.99/89.65                 | PACT在线量化 | 70.63/89.65                 |

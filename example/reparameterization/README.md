# 重参数化

本示例介绍如何对动态图模型进行重参数化训练，示例以常用的MobileNetV1模型为例，介绍如何对其进行DBB重参数化实验，DBB参考自[论文](https://arxiv.org/abs/2103.13425)。


## 分类模型的重参数化训练流程

### 准备数据

在当前目录下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要重参数化的模型

- 对于paddle vision支持的[模型](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/vision/models)：`[lenet, mobilenetv1, mobilenetv2, resnet, vgg]`可以直接使用vision内置的模型定义和ImageNet预训练权重


### 训练命令

- MobileNetV1

  启动命令如下：

   ```bash
  # 单卡训练
  python train.py --model=mobilenet_v1
  # 多卡训练，以0到3号卡为例
  python -m paddle.distributed.launch --gpus="0,1,2,3" train.py
   ```

### 重参数化结果

| 模型        | FP32模型准确率（Top1） | 重参数化方法     | 重参数化模型准确率（Top1） |
| ----------- | --------------------------- | ------------ | --------------------------- |
| MobileNetV1 | 70.99                 | DBB | 72.01               |


使用文档：https://paddlepaddle.github.io/PaddleSlim

# PaddleSlim

PaddleSlim是一个模型压缩工具库，包含模型剪裁、定点量化、知识蒸馏、超参搜索和模型结构搜索等一系列模型压缩策略。

对于业务用户，PaddleSlim提供完整的模型压缩解决方案，可用于图像分类、检测、分割等各种类型的视觉场景。
同时也在持续探索NLP领域模型的压缩方案。另外，PaddleSlim提供且在不断完善各种压缩策略在经典开源任务的benchmark,
以便业务用户参考。

对于模型压缩算法研究者或开发者，PaddleSlim提供各种压缩策略的底层辅助接口，方便用户复现、调研和使用最新论文方法。
PaddleSlim会从底层能力、技术咨询合作和业务场景等角度支持开发者进行模型压缩策略相关的创新工作。


## 功能

- 模型剪裁
  - 通道均匀模型剪裁（uniform pruning)
  - 基于敏感度的模型剪裁
  - 基于进化算法的自动模型剪裁

- 定点量化
  - 在线量化训练（training aware）
  - 离线量化（post training）
  - embedding层log域量化

- 知识蒸馏
  - 支持单进程知识蒸馏
  - 支持多进程分布式知识蒸馏

- 神经网络结构自动搜索（NAS）
  - 支持基于进化算法的轻量神经网络结构自动搜索（Light-NAS）
  - 支持One-Shot网络结构自动搜索（Ont-Shot-NAS）
  - 支持 FLOPS / 硬件延时约束
  - 支持多平台模型延时评估
  - 支持用户自定义搜索算法和搜索空间


## 使用

- [Paddle检测库](https://github.com/PaddlePaddle/PaddleDetection/tree/master/slim)：介绍如何在检测库中使用PaddleSlim。
- [Paddle分割库](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/slim)：介绍如何在分割库中使用PaddleSlim。
- [PaddleLite](https://paddlepaddle.github.io/Paddle-Lite/)：介绍如何使用预测库PaddleLite部署PaddleSlim产出的模型。

## 部分压缩策略效果

### 分类模型

数据: ImageNet2012
模型: MobileNetV1

|压缩策略 |精度(top5/top1) |模型大小|
|---|---|---|
| Baseline|89.54% / 70.91%|17.0M|
| 知识蒸馏(ResNet50)|90.92% / 71.97%|17.0M|
| 知识蒸馏(ResNet50) + int8量化训练 |90.94% / 72.01%|4.8M|
| 剪裁(FLOPS -50%)|89.13% / 69.83%|9.0M|
| 剪裁(FLOPs -50) + int8量化训练|89.11% / 69.20%|2.3M|


### 图像检测模型

|        模型         |        压缩方法         |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积（MB） |
| :-----------------: | :---------------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :------------: |
| MobileNet-V1-YOLOv3 |            -            | Pascal VOC |     8     |      76.2      |      76.7      |      75.3      |       94       |
| MobileNet-V1-YOLOv3 | 知识蒸馏(ResNet34-YOLOv3) | Pascal VOC |     8     |  79.0 (+2.8)   |  78.2 (+1.5)   |  75.5 (+0.2)   |       94       |
| MobileNet-V1-YOLOv3 | 剪裁 FLOPs -52.88% | Pascal VOC |     8     |  77.6 (+1.4)   |  77.7 (+1.0)   |  75.5 (+0.2)   |       30.3       |
| MobileNet-V1-YOLOv3 |            -            |    COCO    |     8     |      29.3      |      29.3      |      27.0      |       95       |
| MobileNet-V1-YOLOv3 | 知识蒸馏(ResNet34-YOLOv3) |    COCO    |     8     |  31.4 (+2.1)   |  30.0 (+0.7)   |  27.1 (+0.1)   |       95       |

### 搜索

数据：ImageNet2012

| -             | 推理耗时 | Top1/Top5准确率 |
|---------------|---------|--------------------|
| MobileNetV2   | 0%      | 71.90% / 90.55%    |
| RK3288  | -23%    | 71.97% / 90.35%    |
| Android cellphone  | -20%    | 72.06% / 90.36%    |
| iPhone 6s   | -17%    | 72.22% / 90.47%    |

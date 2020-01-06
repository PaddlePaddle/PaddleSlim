## 1. 图象分类

数据集：ImageNet1000类

### 1.1 量化

| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型大小（MB） | 下载 |
|:--:|:---:|:--:|:--:|:--:|
|MobileNetV1|-|70.99%/89.68%| xx | [下载链接]() |
|MobileNetV1|quant_post|xx%/xx%| xx | [下载链接]() |
|MobileNetV1|quant_aware|xx%/xx%| xx | [下载链接]() |
| MobileNetV2 | - |72.15%/90.65%| xx | [下载链接]() |
| MobileNetV2 | quant_post |xx%/xx%| xx | [下载链接]() |
| MobileNetV2 | quant_aware |xx%/xx%| xx | [下载链接]() |
|ResNet50|-|76.50%/93.00%| xx | [下载链接]() |
|ResNet50|quant_post|xx%/xx%| xx | [下载链接]() |
|ResNet50|quant_aware|xx%/xx%| xx | [下载链接]() |



### 1.2 剪枝


| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型大小（MB） | FLOPs（M） | arm时延（ms） | P4时延（ms） | 下载 |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
|MobileNetV1|-|70.99%/89.68%| xx | xx | xx | xx | [下载链接]() |
|MobileNetV1|uniform -xx%|xx%/xx%| xx | xx | xx | xx | [下载链接]() |
|MobileNetV1|sensitive -xx%|xx%/xx%| xx | xx | xx | xx | [下载链接]() |
| MobileNetV2 | - |72.15%/90.65%| xx | xx | xx | xx | [下载链接]() |
| MobileNetV2 | uniform -xx% |xx%/xx%| xx | xx | xx | xx | [下载链接]() |
| MobileNetV2 | sensitive -xx% |xx%/xx%| xx | xx | xx | xx | [下载链接]() |
| ResNet34 | - |74.57%/92.14%| xx | xx | xx | xx | [下载链接]() |
| ResNet34 | uniform -xx% |xx%/xx%| xx | xx | xx | xx | [下载链接]() |
| ResNet34 | auto -xx% |xx%/xx%| xx | xx | xx | xx | [下载链接]() |




### 1.3 蒸馏

| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型大小（MB） | 下载 |
|:--:|:---:|:--:|:--:|:--:|
|MobileNetV1|-|70.99%/89.68%| xx | [下载链接]() |
|MobileNetV1|ResNet50_vd<sup>[1](#trans1)</sup> distill|xx%/xx%| xx | [下载链接]() |
| MobileNetV2 | - |72.15%/90.65%| xx | [下载链接]() |
| MobileNetV2 | ResNet50_vd<sup>[1](#trans1)</sup> distill |xx%/xx%| xx | [下载链接]() |
|ResNet50|-|76.50%/93.00%| xx | [下载链接]() |
|ResNet50|ResNet101<sup>[2](#trans2)</sup> distill|xx%/xx%| xx | [下载链接]() |

!!! note "Note"
    <a name="trans1">[1]</a>：[ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar)预训练模型Top-1/Top-5准确率分别为79.12%/94.44%

    带_vd后缀代表开启了Mixup训练，Mixup相关介绍参考[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
    
    <a name="trans2">[2]</a>：[ResNet101](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar)预训练模型Top-1/Top-5准确率分别为77.56%/93.64%

## 2. 目标检测

数据集：Pasacl VOC & COCO 2017 

### 2.1 量化

数据集： COCO 2017

|              模型              |  压缩方法   | 数据集 | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型大小（MB） |     下载     |
| :----------------------------: | :---------: | :----: | :-------: | :------------: | :------------: | :------------: | :------------: | :----------: |
|      MobileNet-V1-YOLOv3       |      -      |  COCO  |     8     |      29.3      |      29.3      |      27.1      |       xx       | [下载链接]() |
|      MobileNet-V1-YOLOv3       | quant_post  |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
|      MobileNet-V1-YOLOv3       | quant_aware |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain |      -      |  COCO  |     8     |      41.4      |       xx       |       xx       |       xx       | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain | quant_post  |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain | quant_aware |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |



数据集：WIDER-FACE



|      模型      |  压缩方法   | Image/GPU | 输入尺寸 | Easy/Medium/Hard  | 模型大小（MB） |     下载     |
| :------------: | :---------: | :-------: | :------: | :---------------: | :------------: | :----------: |
|   BlazeFace    |      -      |     8     |   640    | 0.915/0.892/0.797 |       xx       | [下载链接]() |
|   BlazeFace    | quant_post  |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
|   BlazeFace    | quant_aware |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-Lite |      -      |     8     |   640    | 0.909/0.885/0.781 |       xx       | [下载链接]() |
| BlazeFace-Lite | quant_post  |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-Lite | quant_aware |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-NAS  |      -      |     8     |   640    | 0.837/0.807/0.658 |       xx       | [下载链接]() |
| BlazeFace-NAS  | quant_post  |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-NAS  | quant_aware |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |

### 2.2 剪枝

数据集：Pasacl VOC & COCO 2017

|              模型              |    压缩方法     |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型大小(MB) | FLOPs（M） | arm时延（ms） | P4时延（ms） |     下载     |
| :----------------------------: | :-------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :----------: | :--------: | :-----------: | :----------: | :----------: |
|      MobileNet-V1-YOLOv3       |        -        | Pasacl VOC |     8     |      76.2      |      76.7      |      75.3      |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
|      MobileNet-V1-YOLOv3       | sensitive  -xx% | Pasacl VOC |     8     |       xx       |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
|      MobileNet-V1-YOLOv3       |        -        |    COCO    |     8     |      29.3      |      29.3      |      27.1      |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
|      MobileNet-V1-YOLOv3       | sensitive -xx%  |    COCO    |     8     |       xx       |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
|         R50-dcn-YOLOv3         |        -        |    COCO    |     8     |      39.1      |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
|         R50-dcn-YOLOv3         | sensitive -xx%  |    COCO    |     8     |       xx       |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
|         R50-dcn-YOLOv3         | sensitive -xx%  |    COCO    |     8     |       xx       |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain |        -        |    COCO    |     8     |      41.4      |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain | sensitive -xx%  |    COCO    |     8     |       xx       |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain | sensitive -xx%  |    COCO    |     8     |       xx       |       xx       |       xx       |      xx      |     xx     |      xx       |      xx      | [下载链接]() |

### 2.3 蒸馏

数据集：Pasacl VOC & COCO 2017


|        模型         |                    压缩方法                    |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型大小（MB） |     下载     |
| :-----------------: | :--------------------------------------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :------------: | :----------: |
| MobileNet-V1-YOLOv3 |                       -                        | Pasacl VOC |     8     |      76.2      |      76.7      |      75.3      |       xx       | [下载链接]() |
| MobileNet-V1-YOLOv3 | ResNet34-YOLOv3<sup>[3](#trans3)</sup> distill | Pasacl VOC |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
| MobileNet-V1-YOLOv3 |                       -                        |    COCO    |     8     |      29.3      |      29.3      |      27.1      |       xx       | [下载链接]() |
| MobileNet-V1-YOLOv3 | ResNet34-YOLOv3<sup>[4](#trans4)</sup> distill |    COCO    |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |

!!! note "Note"
    <a name="trans3">[3]</a>：[ResNet34-YOLOv3-VOC]()预训练模型在608/416/320尺寸输入下的Box AP分别为82.6/81.9/80.1
    

    <a name="trans4">[4]</a>：[ResNet34-YOLOv3-COCO]()预训练模型在608/416/320尺寸输入下的Box AP分别为36.2/34.3/31.4

## 3. 图像分割

数据集：Cityscapes

### 3.1 量化

|          模型          |  压缩方法   | mIoU  | 模型大小（MB） |     下载     |
| :--------------------: | :---------: | :---: | :------------: | :----------: |
| DeepLabv3+/MobileNetv1 |      -      | 63.26 |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv1 | quant_post  |  xx   |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv1 | quant_aware |  xx   |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv2 |      -      | 69.81 |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv2 | quant_post  |  xx   |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv2 | quant_aware |  xx   |       xx       | [下载链接]() |

### 3.2 剪枝

|          模型          |  压缩方法  | mIoU  | 模型大小（MB） | FLOPs（M） | arm时延（ms） | P4时延（ms） |     下载     |
| :--------------------: | :--------: | :---: | :------------: | :--------: | :-----------: | :----------: | :----------: |
| DeepLabv3+/MobileNetv2 |     -      | 69.81 |       xx       |     xx     |      xx       |      xx      | [下载链接]() |
| DeepLabv3+/MobileNetv2 | prune -xx% |  xx   |       xx       |     xx     |      xx       |      xx      | [下载链接]() |






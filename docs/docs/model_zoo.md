## 1. 图像分类

数据集：ImageNet1000类

### 1.1 量化

| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型体积（MB） | 下载 |
|:--:|:---:|:--:|:--:|:--:|
|MobileNetV1|FP32 baseline|70.99%/89.68%| xx | [下载链接]() |
|MobileNetV1|quant_post|xx%/xx%| xx | [下载链接]() |
|MobileNetV1|quant_aware|xx%/xx%| xx | [下载链接]() |
| MobileNetV2 | FP32 baseline |72.15%/90.65%| xx | [下载链接]() |
| MobileNetV2 | quant_post |xx%/xx%| xx | [下载链接]() |
| MobileNetV2 | quant_aware |xx%/xx%| xx | [下载链接]() |
|ResNet50|FP32 baseline|76.50%/93.00%| xx | [下载链接]() |
|ResNet50|quant_post|xx%/xx%| xx | [下载链接]() |
|ResNet50|quant_aware|xx%/xx%| xx | [下载链接]() |



### 1.2 剪裁


| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型体积（MB） | GFLOPs | 下载 |
|:--:|:---:|:--:|:--:|:--:|:--:|
| MobileNetV1 |    baseline    |         70.99%/89.68%         |       17       |  1.11  | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) |
| MobileNetV1 |  uniform -50%  | 69.4%/88.66% (-1.59%/-1.02%)  |       9        |  0.56  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_uniform-50.tar) |
| MobileNetV1 | sensitive -30% |  70.4%/89.3% (-0.59%/-0.38%)  |       12       |  0.74  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_sensitive-30.tar) |
| MobileNetV1 | sensitive -50% | 69.8% / 88.9% (-1.19%/-0.78%) |       9        |  0.56  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_sensitive-50.tar) |
| MobileNetV2 |    baseline    |         72.15%/90.65%         |       15       |  0.59  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) |
| MobileNetV2 |  uniform -50%  | 65.79%/86.11% (-6.35%/-4.47%) |       11       | 0.296  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV2_uniform-50.tar) |
|  ResNet34   |    baseline    |         72.15%/90.65%         |       84       |  7.36  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) |
|  ResNet34   |  uniform -50%  | 70.99%/89.95% (-1.36%/-0.87%) |       41       |  3.67  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet34_uniform-50.tar) |
|  ResNet34   |  auto -55.05%  | 70.24%/89.63% (-2.04%/-1.06%) |       33       |  3.31  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet34_auto-55.tar) |




### 1.3 蒸馏

| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型体积（MB） | 下载 |
|:--:|:---:|:--:|:--:|:--:|
| MobileNetV1 |                     student                     |  70.99%/89.68%  |       17       | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) |
|ResNet50_vd|teacher|79.12%/94.44%| 99 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) |
|MobileNetV1|ResNet50_vd<sup>[1](#trans1)</sup> distill|72.77%/90.68% (+1.78%/+1.00%)| 17 | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_distilled.tar) |
| MobileNetV2 |                     student                     |  72.15%/90.65%  |       15       | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) |
| MobileNetV2 |            ResNet50_vd distill             |  74.28%/91.53% (+2.13%/+0.88%)  |       15       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV2_distilled.tar) |
|  ResNet50   |                     student                     |  76.50%/93.00%  |       99       | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) |
|ResNet101|teacher|77.56%/93.64%| 173 | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) |
|  ResNet50   |             ResNet101 distill              |  77.29%/93.65% (+0.79%/+0.65%)  |       99       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet50_distilled.tar) |

!!! note "Note"

    <a name="trans1">[1]</a>：带_vd后缀代表该预训练模型使用了Mixup，Mixup相关介绍参考[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)


## 2. 目标检测

### 2.1 量化

数据集： COCO 2017

|              模型              |   压缩方法    | 数据集 | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积（MB） |     下载     |
| :----------------------------: | :-----------: | :----: | :-------: | :------------: | :------------: | :------------: | :------------: | :----------: |
|      MobileNet-V1-YOLOv3       | FP32 baseline |  COCO  |     8     |      29.3      |      29.3      |      27.1      |       xx       | [下载链接]() |
|      MobileNet-V1-YOLOv3       |  quant_post   |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
|      MobileNet-V1-YOLOv3       |  quant_aware  |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain | FP32 baseline |  COCO  |     8     |      41.4      |       xx       |       xx       |       xx       | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain |  quant_post   |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |
| R50-dcn-YOLOv3 obj365_pretrain |  quant_aware  |  COCO  |     8     |       xx       |       xx       |       xx       |       xx       | [下载链接]() |



数据集：WIDER-FACE



|      模型      |   压缩方法    | Image/GPU | 输入尺寸 | Easy/Medium/Hard  | 模型体积（MB） |     下载     |
| :------------: | :-----------: | :-------: | :------: | :---------------: | :------------: | :----------: |
|   BlazeFace    | FP32 baseline |     8     |   640    | 0.915/0.892/0.797 |       xx       | [下载链接]() |
|   BlazeFace    |  quant_post   |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
|   BlazeFace    |  quant_aware  |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-Lite | FP32 baseline |     8     |   640    | 0.909/0.885/0.781 |       xx       | [下载链接]() |
| BlazeFace-Lite |  quant_post   |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-Lite |  quant_aware  |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-NAS  | FP32 baseline |     8     |   640    | 0.837/0.807/0.658 |       xx       | [下载链接]() |
| BlazeFace-NAS  |  quant_post   |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |
| BlazeFace-NAS  |  quant_aware  |     8     |   640    |     xx/xx/xx      |       xx       | [下载链接]() |

### 2.2 剪裁

数据集：Pasacl VOC & COCO 2017

|              模型              |     压缩方法      |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积(MB) | GFLOPs (608*608) |                             下载                             |
| :----------------------------: | :---------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :----------: | :--------------: | :----------------------------------------------------------: |
|      MobileNet-V1-YOLOv3       |     baseline      | Pascal VOC |     8     |      76.2      |      76.7      |      75.3      |      94      |      40.49       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
|      MobileNet-V1-YOLOv3       | sensitive -52.88% | Pascal VOC |     8     |  77.6 (+1.4)   |   77.7 (1.0)   |  75.5 (+0.2)   |      31      |      19.08       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_v1_voc_prune.tar) |
|      MobileNet-V1-YOLOv3       |     baseline      |    COCO    |     8     |      29.3      |      29.3      |      27.0      |      95      |      41.35       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
|      MobileNet-V1-YOLOv3       | sensitive -51.77% |    COCO    |     8     |  26.0 (-3.3)   |  25.1 (-4.2)   |  22.6 (-4.4)   |      32      |      19.94       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_v1_prune.tar) |
|         R50-dcn-YOLOv3         |     baseline      |    COCO    |     8     |      39.1      |       -        |       -        |     177      |      89.60       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar) |
|         R50-dcn-YOLOv3         | sensitive -9.37%  |    COCO    |     8     |  39.3 (+0.2)   |       -        |       -        |     150      |      81.20       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_prune.tar) |
|         R50-dcn-YOLOv3         | sensitive -24.68% |    COCO    |     8     |  37.3 (-1.8)   |       -        |       -        |     113      |      67.48       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_prune578.tar) |
| R50-dcn-YOLOv3 obj365_pretrain |     baseline      |    COCO    |     8     |      41.4      |       -        |       -        |     177      |      89.60       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365_pretrained_coco.tar) |
| R50-dcn-YOLOv3 obj365_pretrain | sensitive -9.37%  |    COCO    |     8     |  40.5 (-0.9)   |       -        |       -        |     150      |      81.20       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_prune.tar) |
| R50-dcn-YOLOv3 obj365_pretrain | sensitive -24.68% |    COCO    |     8     |  37.8 (-3.3)   |       -        |       -        |     113      |      67.48       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_prune578.tar) |

### 2.3 蒸馏

数据集：Pasacl VOC & COCO 2017


|        模型         |        压缩方法         |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积（MB） |                             下载                             |
| :-----------------: | :---------------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :------------: | :----------------------------------------------------------: |
| MobileNet-V1-YOLOv3 |         student         | Pascal VOC |     8     |      76.2      |      76.7      |      75.3      |       94       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
|   ResNet34-YOLOv3   |         teacher         | Pascal VOC |     8     |      82.6      |      81.9      |      80.1      |      162       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| MobileNet-V1-YOLOv3 | ResNet34-YOLOv3 distill | Pascal VOC |     8     |  79.0 (+2.8)   |  78.2 (+1.5)   |  75.5 (+0.2)   |       94       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |
| MobileNet-V1-YOLOv3 |         student         |    COCO    |     8     |      29.3      |      29.3      |      27.0      |       95       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
|   ResNet34-YOLOv3   |         teacher         |    COCO    |     8     |      36.2      |      34.3      |      31.4      |      163       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| MobileNet-V1-YOLOv3 | ResNet34-YOLOv3 distill |    COCO    |     8     |  31.4 (+2.1)   |  30.0 (+0.7)   |  27.1 (+0.1)   |       95       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |


## 3. 图像分割

数据集：Cityscapes

### 3.1 量化

|          模型          |   压缩方法    | mIoU  | 模型体积（MB） |     下载     |
| :--------------------: | :-----------: | :---: | :------------: | :----------: |
| DeepLabv3+/MobileNetv1 | FP32 baseline | 63.26 |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv1 |  quant_post   |  xx   |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv1 |  quant_aware  |  xx   |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv2 | FP32 baseline | 69.81 |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv2 |  quant_post   |  xx   |       xx       | [下载链接]() |
| DeepLabv3+/MobileNetv2 |  quant_aware  |  xx   |       xx       | [下载链接]() |

### 3.2 剪裁

|   模型    |     压缩方法      |     mIoU      | 模型体积（MB） | GFLOPs |                             下载                             |
| :-------: | :---------------: | :-----------: | :------------: | :----: | :----------------------------------------------------------: |
| fast-scnn |     baseline      |     69.64     |       11       | 14.41  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/fast_scnn_cityscape.tar) |
| fast-scnn | uniform  -17.07%  | 69.58 (-0.06) |      8.5       | 11.95  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/fast_scnn_cityscape_uniform-17.tar) |
| fast-scnn | sensitive -47.60% | 66.68 (-2.96) |      5.7       |  7.55  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/fast_scnn_cityscape_sensitive-47.tar) |


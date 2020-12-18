# Introduction

PaddleSlim is a toolkit for model compression. It contains a collection of compression strategies, such as pruning, fixed point quantization, knowledge distillation, hyperparameter searching and neural architecture search.

PaddleSlim provides solutions of compression on computer vision models, such as image classification, object detection and semantic segmentation. Meanwhile, PaddleSlim Keeps exploring advanced compression strategies for language model. Furthermore, benckmark of compression strategies on some open tasks is available for your reference.

PaddleSlim also provides auxiliary and primitive API for developer and researcher to survey, implement and apply the method in latest papers. PaddleSlim will support developer in ability of framework and technology consulting.

## Features

### Pruning

  - Uniform pruning of convolution
  - Sensitivity-based prunning
  - Automated pruning based evolution search strategy
  - Support pruning of various deep architectures such as VGG, ResNet, and MobileNet.
  - Support self-defined range of pruning, i.e., layers to be pruned.

### Fixed Point Quantization

  - **Training aware**
    - Dynamic strategy: During inference, we quantize models with hyperparameters dynamically estimated from small batches of samples.
    - Static strategy: During inference, we quantize models with the same hyperparameters estimated from training data.
    - Support layer-wise and channel-wise quantization.
  - **Post training**

### Knowledge Distillation

  - **Naive knowledge distillation:** transfers dark knowledge by merging the teacher and student model into the same Program
  - **Paddle large-scale scalable knowledge distillation framework Pantheon:** a universal solution for knowledge distillation, more flexible than the naive knowledge distillation, and easier to scale to the large-scale applications.

    - Decouple the teacher and student models --- they run in different processes in the same or different nodes, and transfer knowledge via TCP/IP ports or local files;
    - Friendly to assemble multiple teacher models and each of them can work in either online or offline mode independently;
    - Merge knowledge from different teachers and make batch data for the student model automatically;
    - Support the large-scale knowledge prediction of teacher models on multiple devices.

### Neural Architecture Search

  - Neural architecture search based on evolution strategy.
  - Support distributed search.
  - One-Shot neural architecture search.
  - Differentiable Architecture Search.
  - Support FLOPs and latency constrained search.
  - Support the latency estimation on different hardware and platforms.

## Performance

### Image Classification

Dataset: ImageNet2012; Model: MobileNetV1;

|Method |Accuracy(baseline: 70.91%) |Model Size(baseline: 17.0M)|
|:---:|:---:|:---:|
| Knowledge Distillation(ResNet50)| **+1.06%** | |
| Knowledge Distillation(ResNet50) + int8 quantization |**+1.10%**| **-71.76%**|
| Pruning(FLOPs-50%) + int8 quantization|**-1.71%**|**-86.47%**|


### Object Detection

#### Dataset: Pascal VOC; Model: MobileNet-V1-YOLOv3

|        Method           | mAP(baseline: 76.2%)         | Model Size(baseline: 94MB)      |
| :---------------------:   | :------------: | :------------:|
| Knowledge Distillation(ResNet34-YOLOv3) | **+2.8%**      |              |
| Pruning(FLOPs -52.88%)        | **+1.4%**      | **-67.76%**   |
|Knowledge DistillationResNet34-YOLOv3)+Pruning(FLOPs-69.57%)| **+2.6%**|**-67.00%**|


#### Dataset: COCO; Model: MobileNet-V1-YOLOv3

|        Method           | mAP(baseline: 29.3%) | Model Size|
| :---------------------:   | :------------: | :------:|
| Knowledge Distillation(ResNet34-YOLOv3) |  **+2.1%**     |-|
| Knowledge Distillation(ResNet34-YOLOv3)+Pruning(FLOPs-67.56%) | **-0.3%** | **-66.90%**|

### NAS

Dataset: ImageNet2012; Model: MobileNetV2

|Device           | Infer time cost | Top1 accuracy(baseline:71.90%) |
|:---------------:|:---------:|:--------------------:|
| RK3288  | **-23%**    | +0.07%    |
| Android cellphone  | **-20%**    | +0.16% |
| iPhone 6s   | **-17%**    | +0.32%  |

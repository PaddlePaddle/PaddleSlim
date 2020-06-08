# 目标检测模型量化教程

教程内容请参考：https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/slim/quantization/README.md


## 示例结果

### 训练策略

- 量化策略`post`为使用静态离线量化方法得到的模型，`aware`为在线量化训练方法得到的模型。

### YOLOv3 on COCO

| 骨架网络         | 预训练权重 | 量化策略 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :--------: | :------: | :------: |:------: | :-----------------------------------------------------: |
| MobileNetV1      |  ImageNet  |   post   |   608    |  27.9   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |   post   |   416    |  28.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |   post   |   320    |  26.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   608    |  28.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_aware.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   416    |  28.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_aware.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   320    |  25.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |   post   |   608    |  35.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_post.tar) |
| ResNet34         |  ImageNet  |  aware   |   608    |  35.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |  aware   |   416    |  33.3   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |  aware   |   320    |  30.3   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   608    |  40.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   416    |  37.5   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   320    |  34.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |

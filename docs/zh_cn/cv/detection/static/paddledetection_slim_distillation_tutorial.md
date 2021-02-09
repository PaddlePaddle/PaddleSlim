# 目标检测模型蒸馏教程

教程内容请参考：https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/slim/distillation/README.md


## 示例结果

### MobileNetV1-YOLO-V3-VOC

| FLOPS |输入尺寸|每张GPU图片个数|推理时间（fps）|Box AP|下载|
|:-:|:-:|:-:|:-:|:-:|:-:|
|baseline|608     |16|104.291|76.2|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|608 |16|106.914|79.0|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar)|
|baseline|416 |16|-|76.7|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|416 |16|-|78.2|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar)|
|baseline|320 |16|-|75.3|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|320 |16|-|75.5|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar)|

> 蒸馏后的结果用ResNet34-YOLO-V3做teacher，4GPU总batch_size64训练90000 iter得到

### MobileNetV1-YOLO-V3-COCO

| FLOPS |输入尺寸|每张GPU图片个数|推理时间（fps）|Box AP|下载|
|:-:|:-:|:-:|:-:|:-:|:-:|
|baseline|608     |16|78.302|29.3|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|608 |16|78.523|31.4|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar)|
|baseline|416 |16|-|29.3|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|416 |16|-|30.0|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar)|
|baseline|320 |16|-|27.0|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|320 |16|-|27.1|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar)|

> 蒸馏后的结果用ResNet34-YOLO-V3做teacher，4GPU总batch_size64训练600000 iter得到

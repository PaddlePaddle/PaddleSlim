# 目标检测模型自动压缩示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始自动压缩](#自动压缩流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1. 简介
本示例将以目标检测模型PP-YOLOE为例，介绍如何使用PaddleDetection中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化蒸馏。

## 2.Benchmark

### PP-YOLOE

| 模型  | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| PP-YOLOE-l | 50.9  |  - | 50.6  |   11.2ms  |   7.7ms   |  **6.7ms**  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/ppyoloe_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar) |
| PP-YOLOE-s |  43.1  |   41.2 |   42.6   |   6.51ms  |   2.77ms   |  **2.12ms**  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/ppyoloe_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_s_quant.tar) |
| PP-YOLOE+ s |  43.7  |   - |   42.7   |   -  |   -   |  -  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/ppyoloe_plus_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar) |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- PP-YOLOE-l模型在Tesla V100的GPU环境下测试，并且开启TensorRT，batch_size=1，包含NMS，测试脚本是[benchmark demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/python)。
- PP-YOLOE-s模型在Tesla T4，TensorRT 8.4.1，CUDA 11.2，batch_size=1，不包含NMS，测试脚本是[cpp_infer_ppyoloe](./cpp_infer_ppyoloe)。

### YOLOv8

| 模型  | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| YOLOv8-s | 44.9 |  43.9 | 44.3  |   9.27ms  |   4.65ms   |  **3.78ms**  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/yolov8_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov8_s_500e_coco_trt_nms_quant.tar) |

**注意：**
- 表格中YOLOv8模型均为带NMS的模型，可直接在TRT中部署，如果需要对齐测试标准，需要测试不带NMS的模型。
- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- 表格中的性能在Tesla T4的GPU环境下测试，并且开启TensorRT，batch_size=1。

### SSD on Pascal VOC
| 模型  | Box AP | ACT量化Box AP | TRT-FP32 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- | :---------------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| SSD-MobileNetv1 |  73.8  |   73.52    |   4.0ms  |  1.7ms  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/ssd_mbv1_voc_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ssd_mobilenet_v1_quant.tar) |
- 测速环境：Tesla T4，TensorRT 8.4.1，CUDA 11.2，batch_size=1，包含NMS.

### RT-DETR

| 模型              | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 |  TRT-INT8  |                           配置文件                           |                           量化模型                           |
| :---------------- | :------- | :---------: | :--------: | :------: | :------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| RT-DETR-R50       | 53.1     |    52.9     |    53.0    | 32.05ms  |  9.12ms  | **6.96ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r50vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r50vd_6x_coco_quant.tar) |
| RT-DETR-R101      | 54.3     |      -      |    54.1    | 54.13ms  | 12.68ms  | **9.20ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r101vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r101vd_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-L | 53.0     |      -      |    52.9    | 26.16ms  |  8.54ms  | **6.65ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_l_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-X | 54.8     |      -      |    54.6    | 49.22ms  | 12.50ms  | **9.24ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_x_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_x_6x_coco_quant.tar) |

- 上表测试环境：Tesla T4，TensorRT 8.6.0，CUDA 11.7，batch_size=1。

| 模型              | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 |  TRT-INT8  |                           配置文件                           |                           量化模型                           |
| :---------------- | :------- | :---------: | :--------: | :------: | :------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| RT-DETR-R50       | 53.1     |    52.9     |    53.0    |  9.64ms  |  5.00ms  | **3.99ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r50vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r50vd_6x_coco_quant.tar) |
| RT-DETR-R101      | 54.3     |      -      |    54.1    | 14.93ms  |  7.15ms  | **5.12ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r101vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r101vd_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-L | 53.0     |      -      |    52.9    |  8.17ms  |  4.77ms  | **4.00ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_l_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-X | 54.8     |      -      |    54.6    | 12.81ms  |  6.97ms  | **5.32ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_x_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_x_6x_coco_quant.tar) |

- 上表测试环境：A10，TensorRT 8.6.0，CUDA 11.6，batch_size=1。
- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- 两个表中的离线量化只量化模型中的conv2d，ACT量化模型中的conv2d和matmul。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.4 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.4
- PaddleDet >= 2.4
- opencv-python

安装paddlepaddle：
```shell
# CPU
python -m pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
#GPU 以ubuntu、CUDA11.6为例
python -m pip install paddlepaddle-gpu==2.6.0.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim：
```shell
pip install paddleslim
```
源码安装(推荐):
git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install

安装paddledet：
```shell
pip install paddledet
```

注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。

#### 3.2 准备数据集

本案例默认以COCO数据进行自动压缩实验，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。

如果数据集为非COCO格式数据，请修改[configs](./configs)中reader配置文件中的Dataset字段。

以PP-YOLOE模型为例，如果已经准备好数据集，请直接修改[./configs/yolo_reader.yml]中`EvalDataset`和`TrainDataset'的`dataset_dir`字段为自己数据集路径即可。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。


根据[PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#8-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) 导出Inference模型，具体可参考下方PP-YOLOE模型的导出示例：
- 下载代码
```
git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型
- 当你使用Paddle Inference但不使用TensorRT时，运行以下命令导出模型(不包含NMS)
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams \
        exclude_post_process=True \
```
PPYOLOE-l模型，包含NMS：如快速体验，可直接下载[PP-YOLOE-l导出模型](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar)
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams \
        trt=True \
```

PPYOLOE-s模型，不包含NMS：如快速体验，可直接下载[PP-YOLOE-s导出模型](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_s_300e_coco.tar)
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams \
        trt=True exclude_post_process=True \
```

YOLOv8-s模型，包含NMS，具体可参考[YOLOv8模型文档](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8), 然后执行：
```shell
python tools/export_model.py \
        -c configs/yolov8/yolov8_s_500e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams \
        trt=True
```

如快速体验，可直接下载[YOLOv8-s导出模型](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov8_s_500e_coco_trt_nms.tar)

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：
注意!!!，ppyoloe_s_qat_dis.yaml中属性include_nms，它默认为False，如果你导出的模型有nms，则将它修改为True。
- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./output/'
```

- 多卡训练：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./rtdetr_hgnetv2_l_6x_coco_quant' --config_path=./configs/rtdetr_hgnetv2_l_qat_dis.yaml
```
0*640

## 4.预测部署

#### 4.1 Paddle Inference 验证性能

量化模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 model.pdmodel 和 model.pdiparams 两个文件 |
| reader_config | eval时模型reader的配置文件路径 |
| image_file | 如果只测试单张图片效果，直接根据image_file指定图片路径 |
| device | 使用GPU或者CPU预测，可选CPU/GPU   |
| use_trt | 是否使用 TesorRT 预测引擎   |
| use_mkldnn | 是否启用```MKL-DNN```加速库，注意```use_mkldnn```与```use_gpu```同时为```True```时，将忽略```enable_mkldnn```，而使用```GPU```预测  |
| cpu_threads | CPU预测时，使用CPU线程数量，默认10  |
| precision | 预测精度，包括`fp32/fp16/int8`  |
| include_nms | 是否包含nms，如果不包含nms，则设置False，如果包含nms，则设置为True  |
| use_dynamic_shape | 是否使用动态shape，如果使用动态shape，则设置为True，否则设置为False  |
| image_shape | 输入图片的大小。这里默认为640,意味着图像将被调整到640*640  |
| trt_calib_mode | 如果模型是通过TensorRT离线量化校准生成的，那么需要将此参数设置为True。|


- TensorRT预测：

环境配置：如果使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，下载地址：[Python预测库](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)
带NMS的
```shell
python paddle_inference_eval.py \
    --model_path=ppyoloe_crn_s_300e_coco \
    --reader_config=configs/yolo_reader.yml \
    --use_trt=True \
    --precision=fp16 \
    --include_nms=True \
    --benchmark=True
```
不带NMS的
```shell
python paddle_inference_eval.py \
    --model_path=ppyoloe_crn_l_300e_coco \
    --reader_config=configs/yolo_reader.yml \
    --use_trt=True \
    --precision=fp16 \
    --include_nms=False \
    --benchmark=True
```
- 原生GPU预测:
```shell
python paddle_inference_eval.py \
    --model_path=ppyoloe_crn_s_300e_coco \
    --reader_config=configs/yolo_reader.yml \
    --device=GPU \
    --precision=fp16 \
    --include_nms=True \
    --benchmark=True
```
- MKLDNN预测：

```shell
python paddle_inference_eval.py \
      --model_path=models/ppyoloe_crn_l_300e_coco_quant \
      --reader_config=configs/yoloe_reader.yml \
      --device=CPU \
      --use_mkldnn=True \
      --cpu_threads=10 \
      --precision=int8
```

- 模型为PPYOLOE，同时不包含NMS，可以使用C++预测demo进行测速：

  直接参考https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/gpu/ppyoloe_crn_l

## 5.FAQ

- 如果想对模型进行离线量化，可进入[Detection模型离线量化示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/detection)中进行实验。

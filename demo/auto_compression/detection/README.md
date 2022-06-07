# 目标检测模型自动压缩示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始自动压缩](#自动压缩流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 测试模型精度](#34-测试模型精度)
  - [3.5 自动压缩并产出模型](#35-自动压缩并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1. 简介
本示例将以目标检测模型PP-YOLOE-l为例，介绍如何使用PaddleDetection中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化蒸馏。


## 2.Benchmark

### PP-YOLOE

| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP32</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| PP-YOLOE-l |  Base模型 | 640*640  |  50.9   |   11.2  |   7.7ms   |  -  |  [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco.tar) |
| PP-YOLOE-l |  量化+蒸馏 | 640*640  |  50.6   |   - |   -   |  6.7ms  |  [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/auto_compression/detection/configs/ppyoloe_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar) |

- mAP的指标均在COCO val2017数据集中评测得到。
- PP-YOLOE模型在Tesla V100的GPU环境下测试，测试脚本是[benchmark demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/python)

### YOLOv5
| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP32</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| YOLOv5s |  Base模型 | 640*640  |  37.4   |   6.0  |   4.9ms   |  -  |  - | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/yolov5s_infer.tar) |
| YOLOv5s |  量化+蒸馏 | 640*640  |  36.5   |   - |   -   |  4.5ms  |  [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/auto_compression/detection/configs/yolov5s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar) |

说明：
- mAP的指标均在COCO val2017数据集中评测得到。
- YOLOv5s模型在Tesla V100的GPU环境下测试，测试脚本是[benchmark demo](./infer.py)
- YOLOv5模型源自[ultralytics/yolov5](https://github.com/ultralytics/yolov5)，通过[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)工具转换YOLOv5预测模型步骤：

(1) 安装X2Paddle的1.3.6以上版本；（pip install x2paddle）

(2) 转换模型：
```
x2paddle --framework=onnx --model=yolov5s.onnx --save_dir=pd_model
cp -r pd_model/inference_model/ yolov5_inference_model
```
即可得到YOLOv5s模型的预测模型（`model.pdmodel` 和 `model.pdiparams`）。如想快速体验，可直接下载上方表格中YOLOv5s的Base预测模型。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim develop版本
- PaddleDet >= 2.4
- opencv-python

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

安装paddleslim：
```shell
https://github.com/PaddlePaddle/PaddleSlim.git
python setup.py install
```

安装paddledet：
```shell
pip install paddledet
```

注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。


#### 3.2 准备数据集

本案例默认以COCO数据进行自动压缩实验，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。

如果数据集为非COCO格式数据，请修改[configs](./configs)中reader配置文件中的Dataset字段。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。


根据[PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#8-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) 导出Inference模型，具体可参考下方PP-YOLOE模型的导出示例：
- 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型

```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams \
        trt=True \
```

**注意**：PP-YOLOE导出时设置`trt=True`旨在优化在TensorRT上的性能，其他模型不需要设置`trt=True`。

或直接下载：
```shell
wget https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco.tar
tar -xf ppyoloe_crn_l_300e_coco.tar
```

#### 3.4. 测试模型精度

使用run.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --eval=True
```

**注意**：TinyPose模型暂不支持精度测试。

#### 3.5 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./output/'
```


## 4.预测部署

可以参考[PaddleDetection部署教程](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy)：
- GPU上量化模型开启TensorRT并设置trt_int8模式进行部署；
- CPU上可参考[X86 CPU部署量化模型教程](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/docs/optimize/paddle_x86_cpu_int8.md)；
- 移动端请直接使用[Paddle Lite Demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/lite)部署。

## 5.FAQ

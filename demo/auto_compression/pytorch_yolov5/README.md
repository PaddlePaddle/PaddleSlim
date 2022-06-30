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

飞桨模型转换工具[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)支持将```Caffe/TensorFlow/ONNX/PyTorch```的模型一键转为飞桨（PaddlePaddle）的预测模型。借助X2Paddle的能力，各种框架的推理模型可以很方便的使用PaddleSlim的自动化压缩功能。

本示例将以[ultralytics/yolov5](https://github.com/ultralytics/yolov5)目标检测模型为例，将PyTorch框架模型转换为Paddle框架模型，再使用ACT自动压缩功能进行自动压缩。本示例使用的自动压缩策略为量化训练。

## 2.Benchmark

| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP16</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| YOLOv5s |  Base模型 | 640*640  |  37.4   |   7.8ms  |   4.3ms   |  -  |  - | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/yolov5s_infer.tar) |
| YOLOv5s |  量化+蒸馏 | 640*640  |  36.5   |   - |   -   |  3.4ms  |  [config](./configs/yolov5s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar) |

说明：
- mAP的指标均在COCO val2017数据集中评测得到。
- YOLOv5s模型在Tesla T4的GPU环境下测试，并且开启TensorRT，测试脚本是[benchmark demo](./paddle_trt_infer.py)。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim develop版本
- PaddleDet >= 2.4
- [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) >= 1.3.6
- opencv-python

（1）安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

（2）安装paddleslim：
```shell
https://github.com/PaddlePaddle/PaddleSlim.git
python setup.py install
```

（3）安装paddledet：
```shell
pip install paddledet
```

注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。

（4）安装X2Paddle的1.3.6以上版本：
```shell
pip install x2paddle
```

#### 3.2 准备数据集

本案例默认以COCO数据进行自动压缩实验，并且依赖PaddleDetection中数据读取模块，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。


#### 3.3 准备预测模型

（1）准备ONNX模型：

可通过[ultralytics/yolov5](https://github.com/ultralytics/yolov5) 官方的[导出教程](https://github.com/ultralytics/yolov5/issues/251)来准备ONNX模型。
```
python export.py --weights yolov5s.pt --include onnx
```

(2) 转换模型：
```
x2paddle --framework=onnx --model=yolov5s.onnx --save_dir=pd_model
cp -r pd_model/inference_model/ yolov5_inference_model
```
即可得到YOLOv5s模型的预测模型（`model.pdmodel` 和 `model.pdiparams`）。如想快速体验，可直接下载上方表格中YOLOv5s的[Base预测模型](https://bj.bcebos.com/v1/paddle-slim-models/detection/yolov5s_infer.tar)。


预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。


#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：

- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/yolov5s_qat_dis.yaml --save_dir='./output/'
```

- 多卡训练：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/yolov5s_qat_dis.yaml --save_dir='./output/'
```

#### 3.5 测试模型精度

使用eval.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/yolov5s_qat_dis.yaml
```

**注意**：要测试的模型路径需要在配置文件中`model_dir`字段下进行修改指定。


## 4.预测部署

- Paddle-TensorRT部署:
使用[paddle_trt_infer.py](./paddle_trt_infer.py)进行部署：
```shell
python paddle_trt_infer.py --model_path=output --image_file=images/000000570688.jpg --benchmark=True --run_mode=trt_int8
```

## 5.FAQ

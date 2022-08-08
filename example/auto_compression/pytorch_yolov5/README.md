# YOLOv5目标检测模型自动压缩示例

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
| YOLOv5s |  Base模型 | 640*640  |  37.4   |   5.95ms  |   2.44ms   |  -  |  - | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx) |
| YOLOv5s |  KL离线量化 | 640*640  |  36.0   |   - |   -   |  1.87ms  |  - | - |
| YOLOv5s |  量化蒸馏训练 | 640*640  |  **36.9**   |   - |   -   |  **1.87ms**  |  [config](./configs/yolov5s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar) |


说明：
- mAP的指标均在COCO val2017数据集中评测得到。
- YOLOv5s模型在Tesla T4的GPU环境下开启TensorRT 8.4.1，batch_size=1， 测试脚本是[cpp_infer](./cpp_infer)。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3
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
pip install paddleslim
```

（3）安装paddledet：
```shell
pip install paddledet
```

注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。

#### 3.2 准备数据集

本案例默认以COCO数据进行自动压缩实验，并且依赖PaddleDetection中数据读取模块，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。

如果已经准备好数据集，请直接修改[./configs/yolov6_reader.yml]中`EvalDataset`的`dataset_dir`字段为自己数据集路径即可。

#### 3.3 准备ONNX预测模型

可通过[ultralytics/yolov5](https://github.com/ultralytics/yolov5) 官方的[导出教程](https://github.com/ultralytics/yolov5/issues/251)来准备ONNX模型。也可以下载准备好的[yolov5s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx)。
```shell
python export.py --weights yolov5s.pt --include onnx
```
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

**注意**：如果要测试量化后的模型，模型路径需要在配置文件中`model_dir`字段下进行修改指定。


## 4.预测部署

#### Paddle-TensorRT C++部署

进入[cpp_infer](./cpp_infer)文件夹内，请按照[C++ TensorRT Benchmark测试教程](./cpp_infer/README.md)进行准备环境及编译，然后开始测试：
```shell
# 编译
bash complie.sh
# 执行
./build/trt_run --model_file yolov5s_quant/model.pdmodel --params_file yolov5s_quant/model.pdiparams --run_mode=trt_int8
```

#### Paddle-TensorRT Python部署:

首先安装带有TensorRT的[Paddle安装包](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)。

然后使用[paddle_trt_infer.py](./paddle_trt_infer.py)进行部署：
```shell
python paddle_trt_infer.py --model_path=output --image_file=images/000000570688.jpg --benchmark=True --run_mode=trt_int8
```

## 5.FAQ

- 如果想测试离线量化模型精度，可执行：
```shell
python post_quant.py --config_path=./configs/yolov5s_qat_dis.yaml
```

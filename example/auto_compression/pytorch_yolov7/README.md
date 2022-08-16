# YOLOv7自动压缩示例

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

本示例将以[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)目标检测模型为例，借助[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)的能力，将PyTorch框架模型转换为Paddle框架模型，再使用ACT自动压缩功能进行模型压缩，压缩后的模型可使用Paddle Inference或者导出至ONNX，利用TensorRT部署。

## 2.Benchmark

| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 模型体积 | 预测时延<sup><small>FP32</small><sup><br><sup> |预测时延<sup><small>FP16</small><sup><br><sup> | 预测时延<sup><small>INT8</small><sup><br><sup> |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| YOLOv7 |  Base模型 | 640*640  |  51.1   | 141MB  |  26.84ms  |   7.44ms   |  -  |  - | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx) |
| YOLOv7 |  离线量化 | 640*640  |  50.2   | 36MB |   - |   -   |  4.55ms  |  - | - |
| YOLOv7 |  ACT量化训练 | 640*640  |  **50.9**   | 36MB |   - |   -   |  **4.55ms**  |  [config](./configs/yolov7_qat_dis.yaml) | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.onnx) |
|  |  |  |  |  |  |  |  |  |
| YOLOv7-Tiny |  Base模型 | 640*640  |  37.3   | 24MB  |  5.06ms  |   2.32ms   |  -  |  - | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7-tiny.onnx) |
| YOLOv7-Tiny |  离线量化 | 640*640  |  -   | 6.1MB  |   - |   -   |  1.68ms  |  - | - |
| YOLOv7-Tiny |  ACT量化训练 | 640*640  |  **37.0**   | 6.1MB  |  - |   -   |  **1.68ms**  |  [config](./configs/yolov7_tiny_qat_dis.yaml) | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_tiny_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_tiny_quant.onnx) |

说明：
- mAP的指标均在COCO val2017数据集中评测得到。
- YOLOv7模型在Tesla T4的GPU环境下开启TensorRT 8.4.1，batch_size=1， 测试脚本是[cpp_infer](./cpp_infer)。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle develop每日版本 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)下载安装）
- PaddleSlim develop 版本

（1）安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

（2）安装paddleslim：
```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```


#### 3.2 准备数据集

本示例默认以COCO数据进行自动压缩实验，可以从[MS COCO官网](https://cocodataset.org)下载[Train](http://images.cocodataset.org/zips/train2017.zip)、[Val](http://images.cocodataset.org/zips/val2017.zip)、[annotation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)。

目录格式如下：
```
dataset/coco/
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   |   ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000580008.jpg
│   |   ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
```

如果是自定义数据集，请按照如上COCO数据格式准备数据。


#### 3.3 准备预测模型

（1）准备ONNX模型：

可通过[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)的导出脚本来准备ONNX模型，具体步骤如下：
```shell
git clone https://github.com/WongKinYiu/yolov7.git
python export.py --weights yolov7-tiny.pt --grid
```

**注意**：目前ACT支持不带NMS模型，使用如上命令导出即可。也可以直接下载我们已经准备好的[yolov7.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov7-tiny.onnx)。

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：

- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/yolov7_tiny_qat_dis.yaml --save_dir='./output/'
```

- 多卡训练：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/yolov7_tiny_qat_dis.yaml --save_dir='./output/'
```

#### 3.5 测试模型精度

修改[yolov7_qat_dis.yaml](./configs/yolov7_qat_dis.yaml)中`model_dir`字段为模型存储路径，然后使用eval.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/yolov7_tiny_qat_dis.yaml
```


## 4.预测部署

#### 导出至ONNX使用TensorRT部署

- 首先安装Paddle2onnx：
```shell
pip install paddle2onnx==1.0.0rc3
```

- 然后将量化模型导出至ONNX：
```shell
paddle2onnx --model_dir output/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 13 \
            --enable_onnx_checker True \
            --save_file yolov7_quant.onnx \
            --deploy_backend tensorrt
```

- 进行测试：
```shell
python yolov7_onnx_trt.py --model_path=yolov7_quant.onnx --image_file=images/000000570688.jpg --precision=int8
```

#### Paddle-TensorRT部署
- C++部署

进入[cpp_infer](./cpp_infer)文件夹内，请按照[C++ TensorRT Benchmark测试教程](./cpp_infer/README.md)进行准备环境及编译，然后开始测试：
```shell
# 编译
bash complie.sh
# 执行
./build/trt_run --model_file yolov7_quant/model.pdmodel --params_file yolov7_quant/model.pdiparams --run_mode=trt_int8
```

- Python部署:

首先安装带有TensorRT的[Paddle安装包](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)。

然后使用[paddle_trt_infer.py](./paddle_trt_infer.py)进行部署：
```shell
python paddle_trt_infer.py --model_path=output --image_file=images/000000570688.jpg --benchmark=True --run_mode=trt_int8
```

## 5.FAQ

- 如果想测试离线量化模型精度，可执行：
```shell
python post_quant.py --config_path=./configs/yolov7_qat_dis.yaml
```

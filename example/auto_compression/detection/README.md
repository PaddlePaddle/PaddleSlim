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
本示例将以目标检测模型PP-YOLOE为例，介绍如何使用PaddleDetection中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化蒸馏。


## 2.Benchmark

### PP-YOLOE

| 模型  | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| PP-YOLOE-l | 50.9  |  - | 50.6  |   11.2ms  |   7.7ms   |  **6.7ms**  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/ppyoloe_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar) |
| PP-YOLOE-s |  43.1  |   41.2 |   42.6   |   6.51ms  |   2.77ms   |  **2.12ms**  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/ppyoloe_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_s_quant.tar) |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- PP-YOLOE-l模型在Tesla V100的GPU环境下测试，并且开启TensorRT，batch_size=1，包含NMS，测试脚本是[benchmark demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/python)。
- PP-YOLOE-s模型在Tesla T4，TensorRT 8.4.1，CUDA 11.2，batch_size=1，不包含NMS，测试脚本是[cpp_infer_ppyoloe](./cpp_infer_ppyoloe)。
## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3
- PaddleDet >= 2.4
- opencv-python

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle==2.3.2
# GPU 以Ubuntu、CUDA 11.2为例
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim：
```shell
pip install paddleslim
```

安装paddledet：
```shell
pip install paddledet
```

注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。

#### 3.2 准备数据集

本案例默认以COCO数据进行自动压缩实验，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。

如果数据集为非COCO格式数据，请修改[configs](./configs)中reader配置文件中的Dataset字段。

以PP-YOLOE模型为例，如果已经准备好数据集，请直接修改[./configs/yolo_reader.yml]中`EvalDataset`的`dataset_dir`字段为自己数据集路径即可。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。


根据[PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#8-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) 导出Inference模型，具体可参考下方PP-YOLOE模型的导出示例：
- 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型

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
        trt=True exclude_nms=True \
```

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：

- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./output/'
```

- 多卡训练：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./output/'
```

#### 3.5 测试模型精度

使用eval.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/ppyoloe_l_qat_dis.yaml
```

**注意**：
- 要测试的模型路径可以在配置文件中`model_dir`字段下进行修改。

## 4.预测部署

- 如果模型包含NMS，可以参考[PaddleDetection部署教程](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy)，GPU上量化模型开启TensorRT并设置trt_int8模式进行部署。

- 模型为PPYOLOE，同时不包含NMS，使用以下预测demo进行部署：
  - Paddle-TensorRT C++部署

  进入[cpp_infer](./cpp_infer_ppyoloe)文件夹内，请按照[C++ TensorRT Benchmark测试教程](./cpp_infer_ppyoloe/README.md)进行准备环境及编译，然后开始测试：
  ```shell
  # 编译
  bash complie.sh
  # 执行
  ./build/trt_run --model_file ppyoloe_s_quant/model.pdmodel --params_file ppyoloe_s_quant/model.pdiparams --run_mode=trt_int8
  ```

  - Paddle-TensorRT Python部署:

  首先安装带有TensorRT的[Paddle安装包](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)。然后使用[paddle_trt_infer.py](./paddle_trt_infer.py)进行部署：
  ```shell
  python paddle_trt_infer.py --model_path=output --image_file=images/000000570688.jpg --benchmark=True --run_mode=trt_int8
  ```

## 5.FAQ

- 如果想测试离线量化模型精度，可执行：
```shell
python post_quant.py --config_path=./configs/ppyoloe_s_qat_dis.yaml
```

# PP-YOLOE模型全量化示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始全量化](#全量化流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 测试模型精度](#34-测试模型精度)
  - [3.5 全量化并产出模型](#35-全量化并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1. 简介
本示例将以目标检测模型PP-YOLOE为例，介绍如何使用PaddleDetection中Inference部署模型进行全量化。本示例使用的全量化策略为全量化加蒸馏。


## 2.Benchmark

| 模型  | 策略 | mAP | TRT-FP32 | TRT-FP16 | TRT-INT8  | 模型  |
| :-------- |:-------- |:--------: | :----------------: | :----------------: | :---------------: | :---------------------: |
| PP-YOLOE-s-416 | Baseline | 39.1   |   -   |  -  |  -  | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_s_no_postprocess_416.tar) |
| PP-YOLOE-s-416 |  量化训练 | 38.5  |   -  |   -   |  -  | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_s_no_postprocess_416_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_s_quant_416_no_postprocess.onnx) |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。

## 3. 全量化流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3.4
- PaddleDet >= 2.5
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
pip install paddleslim
```

安装paddledet：
```shell
pip install paddledet
```
注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。


#### 3.2 准备数据集

本案例默认以COCO数据进行全量化实验，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。

如果数据集为非COCO格式数据，请修改[configs](./configs)中reader配置文件中的Dataset字段。

如果已经准备好数据集，请直接修改[./configs/yoloe_416_reader.yml]中`EvalDataset`的`dataset_dir`字段为自己数据集路径即可。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

根据[PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#8-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) 导出Inference模型，具体可参考下方PP-YOLOE模型的导出示例：
- 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型

注意：PP-YOLOE默认导出640x640输入的模型，如果模型输入需要改为416x416，需要在导出时修改ppdet中[ppyoloe_reader.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/_base_/ppyoloe_reader.yml#L2)的`eval_height`和`eval_width`为416。

包含NMS：
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams \
        trt=True \
```

不包含NMS：
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams \
        trt=True exclude_post_process=True \
```

#### 3.4 全量化并产出模型

全量化示例通过auto_compress.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行全量化。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：

- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python auto_compress.py --config_path=./configs/ppyoloe_s_416_qat_dis.yaml --save_dir='./output/'
```

- 多卡训练：
```
CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 auto_compress.py \
          --config_path=./configs/ppyoloe_s_416_qat_dis.yaml --save_dir='./output/'
```

- 离线量化
```
python post_quant.py --config_path=./configs/ppyoloe_s_416_qat_dis.yaml
```

#### 3.5 测试模型精度

- 使用eval.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/ppyoloe_s_416_qat_dis.yaml
```

**注意**：要测试的模型路径可以在配置文件中`model_dir`字段下进行修改。

- 量化导出ONNX，使用ONNXRuntime测试模型精度：
首先导出onnx量化模型
```
paddle2onnx --model_dir=ptq_out/ --model_filename=model.pdmodel \
            --params_filename=model.pdiparams \
            --save_file=ppyoloe_s_quant_416 \
            --deploy_backend=rknn
```
可以根据不同部署后端设置`--deploy_backend`

然后进行评估：
```shell
python3.7 onnxruntime_eval.py --reader_config=configs/yolo_416_reader.yml --model_path=ppyoloe_s_quant_416_no_postprocess.onnx
```

## 4.预测部署


## 5.FAQ

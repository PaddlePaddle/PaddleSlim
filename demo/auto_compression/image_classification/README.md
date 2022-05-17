# 图像分类模型自动压缩示例

本示例将介绍如何使用PaddleClas中Inference部署模型进行自动压缩（量化训练+蒸馏）。

## Benchmark
- MobileNetV1模型

## 自动压缩结果
| 模型 | 策略 | Top-1 Acc | Latency(ms) SD710 threads=4 | 
|:------:|:------:|:------:|:------:|
| MobileNetV1 | Base模型 | 70.90 | | 39.041 | 
| MobileNetV1 | 量化+蒸馏 | 70.49 | 29.238|


## 环境准备

### 1.准备数据
本案例默认以ImageNet1k数据进行自动压缩实验，如数据集为非ImageNet1k格式数据， 请参考[PaddleClas数据准备文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/data_preparation/classification_dataset.md)。

### 2.准备需要量化的环境
- python >= 3.6
- paddlepaddle >= 2.3

```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

安装paddleslim：
```shell
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```


### 3.准备待压缩的部署模型
如果已经准备好部署的model.pdmodel和model.pdiparams部署模型，跳过此步。
可根据[PaddleClas文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/inference_deployment/export_model.md)导出Inference模型。也可在[PaddleClas预训练模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)中直接获取Inference模型，具体可参考下方获取MobileNetV1模型示例：

```shell
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar -zxvf MobileNetV1_infer.tar
```

## 开始自动压缩

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行量化训练和蒸馏。将数据集路径，压缩参数传入，对模型进行量化训练和蒸馏。

```shell
python -m paddle.distributed.launch run.py \
    --model_dir='MobileNetV1_infer' \
    --model_filename='inference.pdmodel' \
    --params_filename='inference.pdiparams' \
    --save_dir='./save_quant_mobilev1/' \
    --batch_size=128 \
    --config_path='./configs/mobilev1.yaml'\
    --data_dir='ILSVRC2012' 
```

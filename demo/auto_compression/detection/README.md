# 目标检测模型自动压缩

本示例将介绍如何使用PaddleDetection中Inference预测模型进行自动压缩。

## 环境准备

### 1. 准备COCO格式数据

参考[COCO数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md#coco%E6%95%B0%E6%8D%AE)

### 2. 准备需要量化的环境

- PaddlePaddle >= 2.2
- PaddleDet >= 2.4

```shell
pip install paddledet
```

### 3 准备待量化Inference模型
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

### 2.4 测试模型精度

使用[run_main.py](run_main.py)脚本得到模型的mAP：
```
python3.7 run_main.py --config_path=./configs/ppyoloe_l_qat_dist.yaml --eval=True
```

## 开始自动压缩

### 进行量化蒸馏自动压缩
蒸馏量化自动压缩示例通过[run_main.py](run_main.py)脚本启动，会使用接口``paddleslim.auto_compression.AutoCompression``对模型进行量化训练。具体运行命令为：
```
python run_main.py --config_path=./configs/ppyoloe_l_qat_dist.yaml --save_dir='./output/' --devices='gpu'
```

## Benchmark

- PP-YOLOE模型：

| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP32</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| PP-YOLOE-l |  Base模型 | 640*640  |  50.9   |   11.2  |   7.7ms   |  -  |  [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco.tar) |
| PP-YOLOE-l |  量化+蒸馏 | 640*640  |  49.5   |   - |   -   |  6.7ms  |  [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/auto-compression/detection/configs/ppyoloe_l_qat_dist.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco_quant.tar) |

- mAP的指标均在COCO val2017数据集中评测得到。
- PP-YOLOE模型在Tesla V100的GPU环境下测试，测试脚本是[benchmark demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/python)


## 部署

可以参考[PaddleDetection部署教程](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy)：
- GPU上量化模型开启TensorRT并设置trt_int8模式进行部署；
- CPU上可参考[X86 CPU部署量化模型教程](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/docs/optimize/paddle_x86_cpu_int8.md)；
- 移动端请直接使用[Paddle Lite Demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/lite)部署。

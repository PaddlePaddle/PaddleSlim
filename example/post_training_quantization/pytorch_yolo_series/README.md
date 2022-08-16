# YOLO系列离线量化示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.离线量化流程](#离线量化流程)
  - [3.1 准备环境](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 离线量化并产出模型](#34-离线量化并产出模型)
  - [3.5 测试模型精度](#35-测试模型精度)
  - [3.6 提高离线量化精度](#36-提高离线量化精度)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)


本示例将以[ultralytics/yolov5](https://github.com/ultralytics/yolov5)，[meituan/YOLOv6](https://github.com/meituan/YOLOv6) 和 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) YOLO系列目标检测模型为例，将PyTorch框架产出的推理模型转换为Paddle推理模型，使用离线量化功能进行压缩，并使用敏感度分析功能提升离线量化精度。离线量化产出的模型可以用PaddleInference部署，也可以导出为ONNX格式模型文件，并用TensorRT部署。


## 2.Benchmark
| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP16</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| YOLOv5s |  Base模型 | 640*640  |  37.4   |   5.95ms  |   2.44ms   |  -  |  - | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx) |
| YOLOv5s |  KL离线量化 | 640*640  |  36.0   |   - |   -   |  1.87ms  |  -  | - |
|  |  |  |  |  |  |  |  |  |
| YOLOv6s |  Base模型 | 640*640  |  42.4   |   9.06ms  |   2.90ms   |  -  |  - | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx) |
| YOLOv6s |  KL离线量化(量化分析前) | 640*640  |  30.3   |   - |   -   |  1.83ms  |  -  | - |
| YOLOv6s |  KL离线量化(量化分析后) | 640*640  |  39.7   |   - |   -   |  -  |  -  | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_analyzed_ptq.tar) |
|  |  |  |  |  |  |  |  |  |
| YOLOv7 |  Base模型 | 640*640  |  51.1   |   26.84ms  |   7.44ms   |  -  |  - | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx) |
| YOLOv7 |  KL离线量化 | 640*640  |  50.2   |   -  |   -   |  4.55ms  |  - | - |

说明：
- mAP的指标均在COCO val2017数据集中评测得到。

## 3. 离线量化流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim > 2.3版本
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

#### 3.2 准备数据集
本示例默认以COCO数据进行自动压缩实验，可以从 [MS COCO官网](https://cocodataset.org) 下载 [Train](http://images.cocodataset.org/zips/train2017.zip)、[Val](http://images.cocodataset.org/zips/val2017.zip)、[annotation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)。

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

#### 3.3 准备预测模型
（1）准备ONNX模型：

- YOLOv5：可通过[ultralytics/yolov5](https://github.com/ultralytics/yolov5) 官方的[导出教程](https://github.com/ultralytics/yolov5/issues/251)来准备ONNX模型，也可以下载准备好的[yolov5s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx)。

- YOLOv6：可通过[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)的导出脚本来准备ONNX模型，也可以直接下载我们已经准备好的[yolov7.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx)。

- YOLOv7：可通过[meituan/YOLOv6](https://github.com/meituan/YOLOv6)官方的[导出教程](https://github.com/meituan/YOLOv6/blob/main/deploy/ONNX/README.md)来准备ONNX模型，也可以下载已经准备好的[yolov6s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx)。


#### 3.4 离线量化并产出模型
离线量化示例通过post_quant.py脚本启动，会使用接口```paddleslim.quant.quant_post_static```对模型进行量化。配置config文件中模型路径、数据路径和量化相关的参数，配置完成后便可对模型进行离线量化。具体运行命令为：
- YOLOv5

```shell
python post_quant.py --config_path=./configs/yolov5s_ptq.yaml --save_dir=./yolov5s_ptq_out
```

- YOLOv6

```shell
python post_quant.py --config_path=./configs/yolov6s_ptq.yaml --save_dir=./yolov6s_ptq_out
```

- YOLOv7

```shell
python post_quant.py --config_path=./configs/yolov7s_ptq.yaml --save_dir=./yolov7s_ptq_out
```


#### 3.5 测试模型精度

修改 [yolov5s_ptq.yaml](./configs/yolov5s_ptq.yaml) 中`model_dir`字段为模型存储路径，然后使用eval.py脚本得到模型的mAP：

```shell
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/yolov5s_ptq.yaml
```


#### 3.6 提高离线量化精度
本节介绍如何使用量化分析工具提升离线量化精度。离线量化功能仅需使用少量数据，且使用简单、能快速得到量化模型，但往往会造成较大的精度损失。PaddleSlim提供量化分析工具，会使用接口```paddleslim.quant.AnalysisQuant```，可视化展示出不适合量化的层，通过跳过这些层，提高离线量化模型精度。

由于YOLOv6离线量化效果较差，以YOLOv6为例，量化分析工具具体使用方法如下：

```shell
python analysis.py --config_path=./configs/yolov6s_analysis.yaml
```

如下图，经过量化分析之后，可以发现`conv2d_2.w_0`， `conv2d_11.w_0`，`conv2d_15.w_0`， `conv2d_46.w_0`， `conv2d_49.w_0` 这些层会导致较大的精度损失。

<p align="center">
<img src="./images/sensitivity_rank.png" width=849 hspace='10'/> <br />
</p>



对比权重直方分布图后，可以发现量化损失较小的层数值分布相对平稳，数值处于-0.25到0.25之间，而量化损失较大的层数值分布非常极端，绝大部分值趋近于0，且数值处于-0.1到0.1之间，尽管看上去都是正太分布，但大量值为0是不利于量化统计scale值的。

<p align="center">
<img src="./images/hist_compare.png" width=849 hspace='10'/> <br />
</p>


经此分析，在进行离线量化时，可以跳过这些导致精度下降较多的层，可使用 [yolov6s_analyzed_ptq.yaml](./configs/yolov6s_analyzed_ptq.yaml)，然后再次进行离线量化。跳过这些层后，离线量化精度上升9.4个点。

```shell
python post_quant.py --config_path=./configs/yolov6s_analyzed_ptq.yaml --save_dir=./yolov6s_analyzed_ptq_out
```

## 4.预测部署


## 5.FAQ
- 如果想对模型进行自动压缩，可进入[YOLO系列模型自动压缩示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/pytorch_yolo_series)中进行实验。

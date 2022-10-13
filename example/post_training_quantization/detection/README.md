# 目标检测模型离线量化示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始离线量化](#离线量化流程)
  - [3.1 准备环境](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 测试模型精度](#34-测试模型精度)
  - [3.5 离线量化并产出模型](#35-离线量化并产出模型)
  - [3.6 提高离线量化精度](#36-提高离线量化精度)

- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1. 简介
本示例将以目标检测模型PP-YOLOE和PicoDet为例，介绍如何使用PaddleDetection中Inference部署模型，使用离线量化功能进行压缩，并使用敏感度分析功能提升离线量化精度。


## 2.Benchmark

| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP16</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| PP-YOLOE-s |  Base模型 | 640*640  |  43.1   |   11.2ms  |   7.7ms   |    -    |    -   | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_s_300e_coco.tar) |
| PP-YOLOE-s |  离线量化 | 640*640  |  42.6    |     -     |     -     |  6.7ms  |    -   |   [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_s_ptq.tar) |
|  |  |  |  |  |  |  |  |  |
| PicoDet-s |  Base模型 | 416*416  |  32.5   |   -  |   -   |  -  |  - | [Model](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar) |
| PicoDet-s |  离线量化(量化分析前) | 416*416  |  0.0   |   - |   -   |  -  |  -  | - |
| PicoDet-s |  离线量化(量化分析后) | 416*416  |  24.9   |   - |   -   |  -  |  -  | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_ptq.tar) |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。


## 3. 离线量化流程

#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3
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
pip install paddleslim
```

安装paddledet：
```shell
pip install paddledet
```

注：安装PaddleDet的目的是为了直接使用PaddleDetection中的Dataloader组件。

#### 3.2 准备数据集

本案例默认以COCO数据进行离线量化实验，如果自定义COCO数据，或者其他格式数据，请参考[PaddleDetection数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md) 来准备数据。

如果数据集为非COCO格式数据，请修改[configs](./configs)中reader配置文件中的Dataset字段。

以PP-YOLOE模型为例，如果已经准备好数据集，请直接修改[./configs/ppyoloe_s_ptq.yml]中`EvalDataset`的`dataset_dir`字段为自己数据集路径即可。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。


根据[PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#8-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) 导出Inference模型，具体可参考下方PP-YOLOE模型的导出示例：
- 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型


- PPYOLOE-s模型，不包含NMS：如快速体验，可直接下载[PP-YOLOE-s导出模型](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_s_300e_coco.tar)
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams \
        trt=True exclude_nms=True \
```

- PicoDet-s模型，包含NMS：如快速体验，可直接下载[PicoDet-s导出模型](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar)

```shell
python tools/export_model.py -c configs/picodet/picodet_s_416_coco_lcnet.yml \
       -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams \
       --output_dir=output_inference \
```

#### 3.4 离线量化并产出模型

离线量化示例通过post_quant.py脚本启动，会使用接口```paddleslim.quant.quant_post_static```对模型进行量化。配置config文件中模型路径、数据路径和量化相关的参数，配置完成后便可对模型进行离线量化。具体运行命令为：

- PPYOLOE-s：

```
export CUDA_VISIBLE_DEVICES=0
python post_quant.py --config_path=./configs/ppyoloe_s_ptq.yaml --save_dir=./ppyoloe_s_ptq
```

- PicoDet-s：

```
export CUDA_VISIBLE_DEVICES=0
python post_quant.py --config_path=./configs/picodet_s_ptq.yaml --save_dir=./picodet_s_ptq
```


#### 3.5 测试模型精度

使用eval.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/ppyoloe_s_ptq.yaml
```

**注意**：
- 要测试的模型路径可以在配置文件中`model_dir`字段下进行修改。

#### 3.6 提高离线量化精度
本节介绍如何使用量化分析工具提升离线量化精度。离线量化功能仅需使用少量数据，且使用简单、能快速得到量化模型，但往往会造成较大的精度损失。PaddleSlim提供量化分析工具，会使用接口```paddleslim.quant.AnalysisQuant```，可视化展示出不适合量化的层，通过跳过这些层，提高离线量化模型精度。

经过多个实验，包括尝试多种激活算法（avg，KL等）、weight的量化方式（abs_max，channel_wise_abs_max），对PicoDet-s进行离线量化后精度均为0，以PicoDet-s为例，量化分析工具具体使用方法如下：

```shell
python analysis.py --config_path=./configs/picodet_s_analysis.yaml
```


如下图，经过量化分析之后，可以发现`conv2d_1.w_0`， `conv2d_3.w_0`，`conv2d_5.w_0`， `conv2d_7.w_0`， `conv2d_9.w_0` 这些层会导致较大的精度损失，这些层均为主干网络中靠前部分的`depthwise_conv`。

<p align="center">
<img src="./images/picodet_analysis.png" width=849 hspace='10'/> <br />
</p>

在保存的 `activation_distribution.pdf` 中，也可以发现以上这些层的 `activation` 存在较多离群点，导致量化效果较差。
<p align="center">
<img src="./images/act_distribution.png" width=849 hspace='10'/> <br />
</p>

经此分析，在进行离线量化时，可以跳过这些导致精度下降较多的层，可使用 [picodet_s_analyzed_ptq.yaml](./configs/picodet_s_analyzed_ptq.yaml)，然后再次进行离线量化。跳过这些层后，离线量化精度上升24.9个点。

```shell
python post_quant.py --config_path=./configs/picodet_s_analyzed_ptq.yaml --save_dir=./picodet_s_analyzed_ptq_out
```

如想分析之后直接产出符合目标精度的量化模型，可在 `picodet_s_analysis.yaml` 中将`get_target_quant_model`设置为True，并填写 `target_metric`，注意 `target_metric` 不能比原模型精度高。

**加速分析过程**

使用量化分析工具时，因需要逐层量化模型并进行验证，因此过程可能较慢，若想加速分析过程，可以在配置文件中设置 `FastEvalDataset` ，输入一个图片数量较少的annotation文件路径。注意，用少量数据验证的模型精度不一定等于全量数据验证的模型精度，若只需分析时获得不同层量化效果的相对排序，可以使用少量数据集；若要求准确精度，请使用全量验证数据集。如需要全量验证数据，将 `FastEvalDataset` 字段删掉即可。

注：分析之后若需要直接产出符合目标精度的量化模型，demo代码不会使用少量数据集验证，会自动使用全量验证数据。


量化分析工具详细介绍见[量化分析工具介绍](../analysis.md)

## 4.预测部署
预测部署可参考[Detection模型自动压缩示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/detection)

## 5.FAQ

- 如果想对模型进行自动压缩，可进入[Detection模型自动压缩示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/detection)中进行实验。

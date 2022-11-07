
<p align="center">
<h1 align="center">PaddleSlim</h1>
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/Paddle?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleSlim?color=9ea"></a>
    <a href="https://pypi.org/project/PaddleSlim/"><img src="https://img.shields.io/pypi/dm/PaddleSlim?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleSlim?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleSlim?color=ccf"></a>
</p>

PaddleSlim是一个专注于深度学习模型压缩的工具库，提供**低比特量化、知识蒸馏、稀疏化和模型结构搜索**等模型压缩策略，帮助开发者快速实现模型的小型化。

## 产品动态

- 🔥 **2022.08.16：[自动化压缩](example/auto_compression)功能升级**
  - 支持直接加载ONNX模型和Paddle模型导出至ONNX
  - 发布量化分析工具试用版，发布[YOLO系列离线量化工具](example/post_training_quantization/pytorch_yolo_series/)
  - 更新[YOLO-Series自动化压缩模型库](example/auto_compression/pytorch_yolo_series)

  | 模型  | Base mAP<sup>val<br>0.5:0.95 | ACT量化mAP<sup>val<br>0.5:0.95  | 模型体积压缩比 | 预测时延<sup><small>FP32</small><sup><br><sup>  | 预测时延<sup><small>INT8</small><sup><br><sup> | 预测加速比 |
  | :-------- |:-------- |:--------: | :--------: | :---------------------: | :----------------: | :----------------: |
  | PPYOLOE-s | 43.1 | 42.6  | 3.9倍  | 6.51ms  | 2.12ms  | 3.1倍 |
  | YOLOv5s | 37.4   | 36.9  | 3.8倍  | 5.95ms  |  1.87ms | 3.2倍 |
  | YOLOv6s | 42.4   | 41.3 | 3.9倍 |  9.06ms  |   1.83ms   | 5.0倍   |
  | YOLOv7 |  51.1   | 50.9 | 3.9倍 |  26.84ms  |   4.55ms   |  5.9倍  |
  | YOLOv7-Tiny | 37.3   | 37.0 | 3.9倍 | 5.06ms  |   1.68ms   |  3.0倍  |


- 🔥 **2022.07.01: 发布[v2.3.0版本](https://github.com/PaddlePaddle/PaddleSlim/releases/tag/v2.3.0)**

  - 发布[自动化压缩功能](example/auto_compression)
    - 支持代码无感知压缩：开发者只需提供推理模型文件和数据，既可进行离线量化（PTQ）、量化训练（QAT）、稀疏训练等压缩任务。
    - 支持自动策略选择，根据任务特点和部署环境特性：自动搜索合适的离线量化方法,自动搜索最佳的压缩策略组合方式。
    - 发布[自然语言处理](example/auto_compression/nlp)、[图像语义分割](example/auto_compression/semantic_segmentation)、[图像目标检测](example/auto_compression/detection)三个方向的自动化压缩示例。
    - 发布`X2Paddle`模型自动化压缩方案:[YOLOv5](example/auto_compression/pytorch_yolo_series)、[YOLOv6](example/auto_compression/pytorch_yolo_series)、[YOLOv7](example/auto_compression/pytorch_yolo_series)、[HuggingFace](example/auto_compression/pytorch_huggingface)、[MobileNet](example/auto_compression/tensorflow_mobilenet)。
  - 升级量化功能
    - 统一量化模型格式；离线量化支持while op；修复BERT大模型量化训练过慢的问题。
    - 新增7种[离线量化方法](docs/zh_cn/tutorials/quant/post_training_quantization.md), 包括HIST, AVG, EMD, Bias Correction, AdaRound等。
  - 支持半结构化稀疏训练
  - 新增延时预估工具
    - 支持对稀疏化模型、低比特量化模型的性能预估；支持预估指定模型在特定部署环境下 (ARM CPU + Paddle Lite) 的推理性能；提供 SD625、SD710、RK3288 芯片 + Paddle Lite 的预估接口。
    - 提供部署环境自动扩展工具，可以自动增加在更多 ARM CPU 设备上的预估工具。

<details>
<summary>历史更新</summary>

- **2021.11.15: 发布v2.2.0版本**

  - 支持动态图离线量化功能.

- **2021.5.20: 发布V2.1.0版本**

  - 扩展离线量化方法
  - 新增非结构化稀疏
  - 增强剪枝功能
  - 修复OFA功能若干bug

更多信息请参考：[release note](https://github.com/PaddlePaddle/PaddleSlim/releases)

</details>

## 基础压缩功能概览

PaddleSlim支持以下功能，也支持自定义量化、裁剪等功能。
<table>
<tr align="center" valign="bottom">
  <th><a href="https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/tutorials/quant/overview.md">Quantization</a></th>
  <th><a href="https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/tutorials/pruning/overview.md">Pruning</a></th>
  <th><a href="https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/tutorials/nas/overview.md">NAS</a></th>
  <th><a href="https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.0.0/docs/zh_cn/tutorials">Distilling</a></th>
</tr>
<tr valign="top">
  <td>
    <ul>
      <li><a href="docs/zh_cn/api_cn/overview.md#%E5%9C%A8%E7%BA%BF%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83qat">QAT</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#pact">PACT</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#%E9%9D%99%E6%80%81%E7%A6%BB%E7%BA%BF%E9%87%8F%E5%8C%96ptq-static">PTQ Static</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#%E5%8A%A8%E6%80%81%E7%A6%BB%E7%BA%BF%E9%87%8F%E5%8C%96ptq-dynamic">PTQ Dynamic</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#embedding%E9%87%8F%E5%8C%96">Embedding Quant</a></li>
    </ul>
  </td>
  <td>
    <ul>
      <li><a href="docs/zh_cn/api_cn/overview.md#%E6%95%8F%E6%84%9F%E5%BA%A6%E5%89%AA%E6%9E%9D">SensitivityPruner</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#fpgm">FPGMFilterPruner</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#l1norm">L1NormFilterPruner</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#l2norm">**L2NormFilterPruner</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#slimfilter">*SlimFilterPruner</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#optslimfilter">*OptSlimFilterPruner</a></li>
    </ul>
  </td>
  <td>
    <ul>
      <li><a href="docs/zh_cn/api_cn/overview.md#sanas">*Simulate Anneal based NAS</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#rlnas">*Reinforcement Learning based NAS</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#darts">**DARTS</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#pc-darts">**PC-DARTS</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#once-for-all">**Once-for-All</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#hardware-aware-search">*Hardware-aware Search</a></li>
    </ul>
  </td>

  <td>
    <ul>
      <li><a href="docs/zh_cn/api_cn/overview.md#fsp">*FSP</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#dml">*DML</a></li>
      <li><a href="docs/zh_cn/api_cn/overview.md#dk">*DK</a></li>
    </ul>
  </td>
</tr>
</table>

注：
- *表示仅支持静态图，**表示仅支持动态图
- 敏感度裁剪指的是通过各个层的敏感度分析来确定各个卷积层的剪裁率，需要和其他裁剪方法配合使用。

### 多场景效果展示

PaddleSlim在典型视觉和自然语言处理任务上做了模型压缩，并且测试了Nvidia GPU、ARM等设备上的加速情况，这里展示部分模型的压缩效果，详细方案可以参考下面CV和NLP模型压缩方案:

<p align="center">
<img src="docs/images/benchmark.png" height=185 width=849 hspace='10'/> <br />
<strong>表1: 部分场景模型压缩加速情况</strong>
</p>

注:
- YOLOv3: 在移动端SD855上加速3.55倍。
- PP-OCR: 体积由8.9M减少到2.9M, 在SD855上加速1.27倍。
- BERT: 模型参数由110M减少到80M，精度提升的情况下，Tesla T4 GPU FP16计算加速1.47倍。

### 自动压缩效果展示

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/7534971/168805367-f9d1299d-93e3-44d0-84da-870217edeb54.png"/> <br />
<strong>表3: 自动压缩效果</strong>
</p>

### 离线量化效果对比

<p align="center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/7534971/169042883-9ca281ce-19be-4525-a3d2-c54cea4a2cbd.png"/> <br />
<strong>表2: 多种离线量化方法效果对比</strong>
</p>

## 文档教程

## 版本对齐

|  PaddleSlim   | PaddlePaddle   | PaddleLite    |
| :-----------: | :------------: | :------------:|
| 1.0.1         | <=1.7          |       2.7     |
| 1.1.1         | 1.8            |       2.7     |
| 1.2.0         | 2.0Beta/RC     |       2.8     |
| 2.0.0         | 2.0            |       2.8     |
| 2.1.0         | 2.1.0          |       2.8     |
| 2.1.1         | 2.1.1          |       >=2.8   |
| 2.3.0         | 2.3.0          |       >=2.11  |



## 安装

安装最新版本：
```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装指定版本：
```bash
pip install paddleslim==2.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装develop版本：
```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```

### 验证安装

安装完成后您可以使用 python 或 python3 进入 python 解释器，输入import paddleslim, 没有报错则说明安装成功。

### 快速开始

快速开始教程是能基于CIFAR10数据集快速运行起来的简单示例，若您是Paddle官方模型套件用户，请直接使用下方的CV模型压缩或者NLP模型压缩中教程。

- 🔥 [自动压缩](example/auto_compression)
- [量化训练](docs/zh_cn/quick_start/static/quant_aware_tutorial.md)
- [离线量化](docs/zh_cn/quick_start/static/quant_post_static_tutorial.md)
- [结构化剪枝](docs/zh_cn/quick_start/static/pruning_tutorial.md)
- [蒸馏](docs/zh_cn/quick_start/static/distillation_tutorial.md)
- [NAS](docs/zh_cn/quick_start/static/nas_tutorial.md)

### 更多教程

进阶教程详细介绍了每一步的流程，帮助您把相应方法迁移到您自己的模型上。

- 通道剪裁
  - [四种剪裁策略效果对比与应用方法](docs/zh_cn/tutorials/pruning/overview.md)
    - [L1NormFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#l1normfilterpruner)
    - [FPGMFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#fpgmfilterpruner)
    - [SlimFilterFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#slimfilterpruner)
    - [OptSlimFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#optslimfilterpruner)
  - 自定义剪裁策略：[动态图](docs/zh_cn/tutorials/pruning/dygraph/self_defined_filter_pruning.md)

- 低比特量化
  - [三种量化方法介绍与应用](docs/zh_cn/tutorials/quant/overview.md)
    - [量化训练](docs/zh_cn/quick_start/static/quant_aware_tutorial.md)
    - [离线量化](docs/zh_cn/tutorials/quant/static/quant_post_tutorial.md) | [离线量化方法解析](docs/zh_cn/tutorials/quant/post_training_quantization.md)
    - [embedding量化](docs/zh_cn/tutorials/quant/static/embedding_quant_tutorial.md)

- NAS
  - [四种NAS策略介绍和应用](docs/zh_cn/tutorials/nas/overview.md)
    - [Once-For-All](docs/zh_cn/tutorials/nas/dygraph/nas_ofa.md)
    - [SANAS](docs/zh_cn/tutorials/nas/static/sanas_darts_space.md)
    - [RLNAS](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.0.0/demo/nas#rlnas%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2%E7%A4%BA%E4%BE%8B)
    - [DARTS](demo/darts/README.md)

- 蒸馏
  - [知识蒸馏示例](demo/distillation)


#### 推理部署

- [Intel CPU量化部署](demo/mkldnn_quant/README.md)
- [Nvidia GPU量化部署](demo/quant/deploy/TensorRT/README.md)
- [PaddleLite量化部署](docs/zh_cn/deploy/deploy_cls_model_on_mobile_device.md)

### CV模型压缩
本系列教程均基于Paddle官方的模型套件中模型进行压缩，若您不是模型套件用户，更推荐使用快速教程和进阶教程。

- 检测模型压缩
  - 压缩方案
    - [PPDetection-YOLOv3 压缩方案](docs/zh_cn/cv/detection/static/yolov3_slim.md)

  - 方法应用-静态图
    - [蒸馏](docs/zh_cn/cv/detection/static/paddledetection_slim_distillation_tutorial.md)
    - [量化训练](docs/zh_cn/cv/detection/static/paddledetection_slim_quantization_tutorial.md)
    - [模型结构搜索](docs/zh_cn/cv/detection/static/paddledetection_slim_nas_tutorial.md)
    - [剪枝](docs/zh_cn/cv/detection/static/paddledetection_slim_pruing_tutorial.md)
    - [剪枝与蒸馏的结合使用](docs/zh_cn/cv/detection/static/paddledetection_slim_prune_dist_tutorial.md)
    - [卷积层敏感度分析](docs/zh_cn/cv/detection/static/paddledetection_slim_sensitivy_tutorial.md)

  - 方法应用-动态图
    - [剪枝](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0-rc/dygraph/configs/slim#%E5%89%AA%E8%A3%81)
    - [量化训练](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0-rc/dygraph/configs/slim#%E9%87%8F%E5%8C%96)

- 分割模型压缩

  - 压缩方案

  - 方法应用-静态图
    - [蒸馏](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.8.0/slim/distillation)
    - [量化训练](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.8.0/slim/quantization)
    - [模型结构搜索](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.8.0/slim/nas)
    - [剪枝](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.8.0/slim/prune)

  - 方法应用-动态图
    - [剪枝](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/slim#%E6%A8%A1%E5%9E%8B%E8%A3%81%E5%89%AA)
    - [量化训练](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/slim#%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96)

- OCR模型压缩

  - 压缩方案
    - [3.5M模型压缩方案](docs/zh_cn/cv/ocr/static/3.5M_slim.md)

  - 方法应用-静态图
    - [量化训练](https://github.com/PaddlePaddle/PaddleOCR/tree/release/1.1/deploy/slim/quantization)
    - [剪枝](https://github.com/PaddlePaddle/PaddleOCR/tree/release/1.1/deploy/slim/prune)

  - 方法应用-动态图
    - [剪枝](https://github.com/PaddlePaddle/PaddleOCR/tree/develop/deploy/slim/prune)
    - [量化训练](https://github.com/PaddlePaddle/PaddleOCR/tree/develop/deploy/slim/quantization)


### NLP模型压缩

- [PaddleNLP-BERT](docs/zh_cn/nlp/paddlenlp_slim_ofa_tutorial.md)
- [ERNIE-ERNIE](docs/zh_cn/nlp/ernie_slim_ofa_tutorial.md)

### API文档

- [动态图](docs/zh_cn/api_cn/dygraph)
- [静态图](docs/zh_cn/api_cn/static)

### [FAQ](docs/zh_cn/FAQ/quantization_FAQ.md)

#### 1. 量化训练或者离线量化后的模型体积为什么没有变小？
答：这是因为量化后保存的参数是虽然是int8范围，但是类型是float。这是因为Paddle训练前向默认的Kernel不支持INT8 Kernel实现，只有Paddle Inference TensorRT的推理才支持量化推理加速。为了方便量化后验证量化精度，使用Paddle训练前向能加载此模型，默认保存的Float32类型权重，体积没有发生变换。


#### 2. macOS + Python3.9环境或者Windows环境下, 安装出错, "command 'swig' failed"

答: 请参考https://github.com/PaddlePaddle/PaddleSlim/issues/1258

## 许可证书

本项目的发布受[Apache 2.0 license](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/LICENSE)许可认证。

## 贡献代码

我们非常欢迎你可以为PaddleSlim提供代码，也十分感谢你的反馈。

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> 技术交流

- 如果你发现任何PaddleSlim存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSlim/issues)给我们提issues。

- 欢迎加入PaddleSlim 微信技术交流群

 <div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/199486336-11d661a7-6cbd-47b1-823c-3e4ac38bb7d5.jpg"  width = "225" height = "225" />
  </div>

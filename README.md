# PaddleSlim

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddleslim.readthedocs.io/en/latest/)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddleslim.readthedocs.io/zh_CN/latest/)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## 简介

PaddleSlim是一个专注于深度学习模型压缩的工具库，提供**剪裁、量化、蒸馏、和模型结构搜索**等模型压缩策略，帮助用户快速实现模型的小型化。

## 版本对齐

|  PaddleSlim   | PaddlePaddle   | PaddleLite    | 备注        |
| :-----------: | :------------: | :------------:| :----------:|
| 1.0.1         | <=1.7          |       2.7     | 支持静态图  |
| 1.1.1         | 1.8            |       2.7     | 支持静态图  |
| 1.2.0         | 2.0Beta/RC     |       2.8     | 支持静态图  |
| 2.0.0         | 2.0            |       2.8     | 支持动态图和静态图  |


## 安装

安装最新版本：
```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装指定版本：
```bash
pip install paddleslim=2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 最近更新

2021.2.5： 发布V2.0.0版本，新增支持动态图，新增OFA压缩功能，优化剪枝功能。
2020.9.16:  发布V1.2.0版本，新增PACT量化训练功能，新增DML(互蒸馏功能)，修复部分剪裁bug，加强对depthwise_conv2d的剪裁能力，优化剪裁和量化API的易用性和灵活性。

更多信息请参考：[release note](https://github.com/PaddlePaddle/PaddleSlim/releases)

## 功能概览

PaddleSlim支持以下功能，也支持自定义量化、裁剪等功能。
<table>
<tr align="center" valign="bottom">
  <th>Quantization</th>
  <th>Pruning</th>
  <th>NAS</th>
  <th>Distilling</th>
</tr>
<tr valign="top">
  <td>
    <ul>
      <li>QAT</li>
      <li>PACT</li>
      <li>PTQ-Static</li>
      <li>PTQ-Dynamic</li>
      <li>Embedding Quant</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>SensitivityPruner</li>
      <li>FPGMFilterPruner</li>
      <li>L1NormFilterPruner</li>
      <li>L2NormFilterPruner</li>
      <li>*SlimFilterPruner</li>
      <li>*OptSlimFilterPruner</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>*Simulate Anneal based NAS</li>
      <li>*Reinforcement Learning based NAS</li>
      <li>**DARTS</li>
      <li>**PC-DARTS</li>
      <li>**Once-for-All</li>
      <li>*Hardware-aware Search</li>
    </ul>
  </td>

  <td>
    <ul>
      <li>*FSP</li>
      <li>*DML</li>
      <li>*DK for YOLOv3</li>
    </ul>
  </td>
</tr>
</table>

注：*表示仅支持静态图，**表示仅支持动态图

### 效果展示

PaddleSlim在典型视觉和自然语言处理任务上做了模型压缩，并且测试了Nvidia GPU、ARM等设备上的加速情况，这里展示部分模型的压缩效果，详细方案可以参考下面CV和NLP模型压缩方案:

<p align="center">
<img src="docs/images/benchmark.png" height=185 width=849 hspace='10'/> <br />
<strong>表1: 部分模型压缩加速情况</strong>
</p>

注:
- YOLOv3: 在移动端SD855上加速3.55倍。
- PP-OCR: 体积由8.9M减少到2.9M, 在SD855上加速1.27倍。
- BERT: 模型参数由110M减少到80M，精度提升的情况下，Tesla T4 GPU FP16计算加速1.47倍。

## 文档教程

### 快速开始

- 量化训练 - [动态图](docs/zh_cn/quick_start/dygraph/dygraph_quant_aware_training_tutorial.md) | [静态图](docs/zh_cn/quick_start/static/quant_aware_tutorial.md)
- 离线量化 - [动态图](docs/zh_cn/quick_start/dygraph/dygraph_quant_post_tutorial.md) | [静态图](docs/zh_cn/quick_start/static/quant_post_static_tutorial.md)
- 剪裁 - [动态图](docs/zh_cn/quick_start/dygraph/dygraph_pruning_tutorial.md) | [静态图](docs/zh_cn/quick_start/static/pruning_tutorial.md)
- 蒸馏 - [静态图](docs/zh_cn/quick_start/static/distillation_tutorial.md)
- NAS - [静态图](docs/zh_cn/quick_start/static/nas_tutorial.md)

### 进阶教程

- 通道剪裁
  - [四种剪裁策略效果对比与应用方法](docs/zh_cn/tutorials/pruning/overview.md)
    - [L1NormFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#l1normfilterpruner)
    - [FPGMFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#fpgmfilterpruner)
    - [SlimFilterFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#slimfilterpruner)
    - [OptSlimFilterPruner](docs/zh_cn/tutorials/pruning/overview.md#optslimfilterpruner)
  - 剪裁功能详解: [动态图](docs/zh_cn/tutorials/pruning/dygraph/filter_pruning.md) | [静态图](docs/zh_cn/tutorials/pruning/static/image_classification_sensitivity_analysis_tutorial.md)
  - 自定义剪裁策略：[动态图](docs/zh_cn/tutorials/pruning/dygraph/self_defined_filter_pruning.md)

- 低比特量化
  - [三种量化方法介绍与应用](docs/zh_cn/tutorials/quant/overview.md)
    - 量化训练：[动态图](docs/zh_cn/tutorials/quant/dygraph/quant_aware_training_tutorial.md) | [静态图](docs/zh_cn/quick_start/static/quant_aware_tutorial.md)
    - 离线量化：[动态图](docs/zh_cn/tutorials/quant/dygraph/dygraph_quant_post_tutorial.md) | [静态图](docs/zh_cn/tutorials/quant/static/quant_post_tutorial.md)
    - embedding量化：[静态图](docs/zh_cn/tutorials/quant/static/embedding_quant_tutorial.md)

- NAS
  - [四种NAS策略介绍和应用](docs/zh_cn/tutorials/nas/overview.md)

- 蒸馏
  - [知识蒸馏示例](demo/distillation)


#### 推理部署

- [Intel CPU量化部署](demo/mkldnn_quant/README.md)
- [Nvidia GPU量化部署](demo/quant/deploy/TensorRT/README.md)
- [PaddleLite量化部署](docs/zh_cn/deploy/deploy_cls_model_on_mobile_device.md)

### CV模型压缩

- 检测模型压缩(基于PaddleDetection)
  - 压缩方案
    - YOLOv3 3.5倍加速方案: 文档整理中...
  - 方法应用-静态图
    - [在COCO和VOC上蒸馏MobileNetV1-YOLOv3](docs/zh_cn/cv/detection/static/paddledetection_slim_distillation_tutorial.md)
    - [MobileNetV1-YOLOv3低比特量化训练](docs/zh_cn/cv/detection/static/paddledetection_slim_quantization_tutorial.md)
    - [人脸检测模型小模型结构搜索](docs/zh_cn/cv/detection/static/paddledetection_slim_nas_tutorial.md)
    - [剪枝](docs/zh_cn/cv/detection/static/paddledetection_slim_pruing_tutorial.md)
    - [剪枝与蒸馏的结合使用](docs/zh_cn/cv/detection/static/paddledetection_slim_prune_dist_tutorial.md)
    - [卷积层敏感度分析](docs/zh_cn/cv/detection/static/paddledetection_slim_sensitivy_tutorial.md)
  - 方法应用-动态图
    - 文档整理中...

- 分割模型压缩(基于PaddleSeg)

  - 压缩方案
    - 方案建设中...

  - 方法应用-静态图
    - 文档整理中...

  - 方法应用-动态图
    - 文档整理中...

- [OCR模型压缩(基于PaddleOCR)]()

  - 压缩方案
    - 3.5M模型压缩方案: 文档整理中...

  - 方法应用-静态图
    - [低比特量化训练](https://github.com/PaddlePaddle/PaddleOCR/tree/release/1.1/deploy/slim/quantization)
    - [剪枝](https://github.com/PaddlePaddle/PaddleOCR/tree/release/1.1/deploy/slim/prune)

  - 方法应用-动态图
    - 文档整理中...


### NLP模型压缩

- [BERT](docs/zh_cn/nlp/paddlenlp_slim_ofa_tutorial.md)
- [ERNIE](docs/zh_cn/nlp/ernie_slim_ofa_tutorial.md)

### API文档

- [动态图](docs/zh_cn/api_cn/dygraph)
- [静态图](docs/zh_cn/api_cn/static)

### [FAQ]()

## 许可证书

本项目的发布受[Apache 2.0 license](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/LICENSE)许可认证。

## 贡献代码

我们非常欢迎你可以为PaddleSlim提供代码，也十分感谢你的反馈。

## 欢迎加入PaddleSlim技术交流群

请添加微信公众号"AIDigest"，备注“压缩”，飞桨同学会拉您进入微信交流群。

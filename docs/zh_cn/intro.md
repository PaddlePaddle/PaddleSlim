# 简介

PaddleSlim是一个专注于深度学习模型压缩的工具库，提供**剪裁、量化、蒸馏、和模型结构搜索**等模型压缩策略，帮助用户快速实现模型的小型化。

## 版本对齐

|  PaddleSlim   | PaddlePaddle   | PaddleLite    | 备注        |
| :-----------: | :------------: | :------------:| :----------:|
| 1.0.1         | <=1.7          |       2.7     | 支持静态图  |
| 1.1.1         | 1.8            |       2.7     | 支持静态图  |
| 1.2.0         | 2.0Beta/RC     |       2.8     | 支持静态图  |
| 2.0.0         | 2.0            |       2.8     | 支持动态图和静态图  |


## 最近更新

- 2021.2.5： 发布V2.0.0版本，新增支持动态图，新增OFA压缩功能，优化剪枝功能。

- 2020.9.16:  发布V1.2.0版本，新增PACT量化训练功能，新增DML(互蒸馏功能)，修复部分剪裁bug，加强对depthwise_conv2d的剪裁能力，优化剪裁和量化API的易用性和灵活性。

## 功能概览

PaddleSlim支持以下功能，也支持自定义量化、裁剪等功能。
<table border=1>
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
<img src="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/images/benchmark.png?raw=true" height=185 width=849 hspace='10'/> <br />
<strong>表1: 部分模型压缩加速情况</strong>
</p>

注:
- YOLOv3: 在移动端SD855上加速3.55倍。
- PP-OCR: 体积由8.9M减少到2.9M, 在SD855上加速1.27倍。
- BERT: 模型参数由110M减少到80M，精度提升的情况下，Tesla T4 GPU FP16计算加速1.47倍。

## 许可证书

本项目的发布受[Apache 2.0 license](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/LICENSE)许可认证。

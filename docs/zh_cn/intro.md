

# PaddleSlim简介

PaddleSlim是PaddlePaddle框架的一个子模块，主要用于压缩图像领域模型。在PaddleSlim中，不仅实现了目前主流的网络剪枝、量化、蒸馏三种压缩策略，还实现了超参数搜索和小模型网络结构搜索功能。在后续版本中，会添加更多的压缩策略，以及完善对NLP领域模型的支持。

## 功能

- 模型剪裁
  - 支持通道均匀模型剪裁（uniform pruning)
  - 基于敏感度的模型剪裁
  - 基于进化算法的自动模型剪裁三种方式

- 量化训练
  - 在线量化训练（training aware）
  - 离线量化（post training）
  - 支持对权重全局量化和Channel-Wise量化

- 蒸馏

- 轻量神经网络结构自动搜索（Light-NAS）
  - 支持基于进化算法的轻量神经网络结构自动搜索（Light-NAS）
  - 支持 FLOPS / 硬件延时约束
  - 支持多平台模型延时评估

# BinaryDM in PaddlePaddle

## 1. 简介

本示例介绍了一种权重二值化的扩散模型的训练方法。通过可学习多基二值化器和低秩表示模仿来增强二值扩散模型的表征能力并提高优化表现，能支持将扩散模型应用于极限资源任务场景中。

技术详情见论文 [BinaryDM: Accurate Weight Binarization for Efficient Diffusion Models](https://arxiv.org/pdf/2404.05662v4)

![binarydm](.\imgs\binarydm.png)

## 2.训练

### 2.1 环境准备

- paddlepaddle>=2.0.1 (paddlepaddle-gpu>=2.0.1)
- visualdl
- lmdb

### 2.2 启动训练

```
python main_binarydm.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni
```

## 致谢

本实现源于下列开源仓库:

- [https://github.com/Xingyu-Zheng/BinaryDM](https://github.com/Xingyu-Zheng/BinaryDM) (official implementation of BinaryDM).
- [https://openi.pcl.ac.cn/iMon/ddim-paddle](https://openi.pcl.ac.cn/iMon/ddim-paddle) (PaddlePaddle version for DDIM).
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (code structure).

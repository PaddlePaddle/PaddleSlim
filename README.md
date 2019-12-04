

# PaddleSlim

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


## 安装

安装PaddleSlim前，请确认已正确安装Paddle1.6版本或更新版本。Paddle安装请参考：[Paddle安装教程](https://www.paddlepaddle.org.cn/install/quick)。


- 安装develop版本


```
git clone http://gitlab.baidu.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```

- 安装官方发布的最新版本

```
pip install paddleslim -i https://pypi.org/simple
```

- 安装历史版本

请点击[pypi.org](https://pypi.org/project/paddleslim/#history)查看可安装历史版本。

## 使用

[API文档]()
[示例]()
[模型库]()
[Paddle检测库]()
[Paddle分割库]()
[PaddleLite]()

以上文档说明如下：

- [API文档]()：API使用介绍，包括[蒸馏]()、[剪裁]()、[量化]()和[模型结构搜索]()。
- [示例]()：基于mnist和cifar10等简单分类任务的模型压缩示例，您可以通过该部分快速体验和了解PaddleSlim的功能。
- [模型库]()：经过压缩的分类、检测、语义分割模型，包括权重文件、网络结构文件和性能数据。
- [Paddle检测库]()：介绍如何在检测库中使用PaddleSlim。
- [Paddle分割库]()：介绍如何在分割库中使用PaddleSlim。
- [PaddleLite]()：介绍如何使用预测库PaddleLite部署PaddleSlim产出的模型。

## 贡献与反馈

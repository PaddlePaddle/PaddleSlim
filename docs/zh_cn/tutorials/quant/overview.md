# Overview

## 图像分类INT8量化模型在CPU上的部署和预测
PaddleSlim主要包含三种量化方法：量化训练(Quant Aware Training, QAT)、动态离线量化(Post Training Quantization Dynamic, PTQ Dynamic)、静态离线量化(Post Training Quantization Static, PTQ Static)。

- [量化训练](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/quant/static/quant_aware_tutorial.html) 量化训练让模型感知量化运算对模型精度带来的影响，通过finetune训练降低量化误差。
- [动态离线量化](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/quant/static/quant_post_tutorial.html) 动态离线量化仅将模型中特定算子的权重从FP32类型映射成INT8/16类型。
- [静态离线量化](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/quant/static/quant_post_tutorial.html) 静态离线量化使用少量无标签校准数据，采用KL散度等方法计算量化比例因子。

除此之外，PaddleSlim还有一种对embedding层量化的方法，将网络中embedding层参数从float32类型量化到int8类型。

- [Embedding量化](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/quant/static/embedding_quant_tutorial.html) Embedding量化仅将embedding参数从float32类型映射到int8类型，可以降低embedding参数体积。

下图展示了如何根据需要选择模型量化方法

![模型量化算法选择](https://user-images.githubusercontent.com/52520497/95644539-e7f23500-0ae9-11eb-80a8-596cfb285e17.png)

下表综合对比了模型量化方法的使用条件、易用性、精度损失和预期收益。

![模型量化算法对比](https://user-images.githubusercontent.com/52520497/95644609-59ca7e80-0aea-11eb-8897-208d7ccd5af1.png)

|             量化方法             |                           API接口                            |                  功能                  |                  经典适用场景                   |
| :------------------------------: | :----------------------------------------------------------: | :------------------------------------: | :---------------------------------------------: |
|          在线量化 (QAT)          | 动态图：`paddleslim.QAT`; 静态图：`paddleslim.quant.quant_aware` | 通过finetune训练将模型量化误差降到最小 | 对量化敏感的场景、模型，例如目标检测、分割, OCR |
|    静态离线量化 (PTQ Static)     |             `paddleslim.quant.quant_post_static`             |      通过少量校准数据得到量化模型      |      对量化不敏感的场景，例如图像分类任务       |
|    动态离线量化 (PTQ Dynamic)    |            `paddleslim.quant.quant_post_dynamic`             |         仅量化模型的可学习权重         |   模型体积大、访存开销大的模型，例如BERT模型    |
| Embedding量化（Quant Embedding） |              `paddleslim.quant.quant_embedding`              |       仅量化模型的Embedding参数        |            任何包含Embedding层的模型            |

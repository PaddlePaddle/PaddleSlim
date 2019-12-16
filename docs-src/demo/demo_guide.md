
## [蒸馏](../demo/distillation/distillation_demo.py)

蒸馏demo默认使用ResNet50作为teacher网络，MobileNet作为student网络，此外还支持将teacher和student换成[models目录](../demo/models)支持的任意模型。

demo中对teahcer模型和student模型的一层特征图添加了l2_loss的蒸馏损失函数，使用时也可根据需要选择fsp_loss, soft_label_loss以及自定义的loss函数。

训练默认使用的是cifar10数据集，piecewise_decay学习率衰减策略，momentum优化器进行120轮蒸馏训练。使用者也可以简单地用args参数切换为使用ImageNet数据集，cosine_decay学习率衰减策略等其他训练配置。

## 量化

### [量化训练demo文档](./quant_aware_demo.md)
### [离线量化demo文档](./quant_post_demo.md)
### [Embedding量化demo文档](./quant_embedding_demo.md)

## NAS

### [NAS示例](./nas_demo.md)

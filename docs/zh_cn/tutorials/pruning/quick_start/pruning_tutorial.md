#  图像分类模型通道剪裁-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的卷积通道剪裁接口](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/dygraph_docs)。
该示例包含以下步骤：

1. 导入依赖
2. 构建模型和数据集
3. 进行预训练
4. 剪裁
5. 训练剪裁后的模型

以下章节依次次介绍每个步骤的内容。

## 1. 导入依赖

请确认已正确安装Paddle，版本依赖关系可见[PaddleSlim Rep主页](https://github.com/PaddlePaddle/PaddleSlim)。然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import paddle.vision.models as models
from paddle.static import InputSpec as Input
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from paddleslim.dygraph import L1NormFilterPruner
```

## 2. 构建网络和数据集

该章节构造一个用于对CIFAR10数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[3, 32, 32]`，输出类别数为10。
为了方便展示示例，我们使用Paddle提供的[预定义分类模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/models/mobilenetv1/MobileNetV1_cn.html#mobilenetv1)和[高层API](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/tutorial/quick_start/high_level_api/high_level_api.html)，执行以下代码构建分类模型：

```
net = models.mobilenet_v1(pretrained=False, scale=1.0, num_classes=10)
inputs = [Input([None, 3, 32, 32], 'float32', name='image')]
labels = [Input([None, 1], 'int64', name='label')]
optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=net.parameters())
model = paddle.Model(net, inputs, labels)
model.prepare(
        optimizer,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])

val_dataset = Cifar10(mode='test', transform=transform)
train_dataset = Cifar10(mode='train', transform=transform)
```

## 3. 进行预训练

对模型进行预训练，为之后的裁剪做准备。
执行以下代码对模型进行预训练
```
model.fit(train_dataset, epochs=2, batch_size=128, verbose=1)
```


## 4. 剪裁卷积层通道

### 4.1 计算剪裁之前的FLOPs

```
FLOPs = paddle.flops(net, input_size=[1, 3, 32, 32], print_detail=True)
```

### 4.2 剪裁

对网络模型两个不同的网络层按照参数名分别进行比例为50%，60%的裁剪。
代码如下所示：

```
pruner = L1NormFilterPruner(net, [1, 3, 32, 32])
pruner.prune_vars({'conv2d_22.w_0':0.5, 'conv2d_20.w_0':0.6}, axis=0)
```

以上操作会按照网络结构中不同网路层的冗余程度对网络层进行不同程度的裁剪并修改网络模型结构。

### 4.3 计算剪裁之后的FLOPs

```
FLOPs = paddle.flops(net, input_size=[1, 3, 32, 32], print_detail=True)
```

## 5. 训练剪裁后的模型

### 5.1 评估裁剪后的模型

对模型进行裁剪会导致模型精度有一定程度下降。
以下代码评估裁剪后模型的精度：

```
model.evaluate(val_dataset, batch_size=128, verbose=1)
```

### 5.2 对模型进行微调
对模型进行finetune会有助于模型恢复原有精度。
以下代码对裁剪过后的模型进行评估后执行了一个`epoch`的微调，再对微调过后的模型重新进行评估：

```
model.fit(train_dataset, epochs=1, batch_size=128, verbose=1)
model.evaluate(val_dataset, batch_size=128, verbose=1)
```

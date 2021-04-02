#  量化训练

量化训练要解决的问题是将FP32浮点数量化成INT8整数进行存储和计算，通过在训练中建模量化对模型的影响，降低量化误差。

PaddleSlim使用的是模拟量化训练方案，一般模拟量化需要先对网络计算图进行一定的处理，先在需要量化的算子前插入量化-反量化节点，再经过finetune训练减少量化运算带来的误差，降低量化模型的精度损失。

下面该教程将以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的模型量化接口]()。

> 注意：目前动态图量化训练还不支持有控制流逻辑的模型，如果量化训练中出现Warning，推荐使用静态图量化训练功能。

该示例包含以下步骤：

1. 导入依赖
2. 构建模型和数据集
3. 进行预训练
4. 量化训练
5. 导出预测模型

以下章节依次次介绍每个步骤的内容。

## 1. 导入依赖

请参考PaddleSlim安装文档，安装正确的Paddle和PaddleSlim版本，然后按以下方式导入Paddle和PaddleSlim:

```python
import paddle
import paddle.vision.models as models
from paddle.static import InputSpec as Input
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from paddleslim.dygraph.quant import QAT
```

## 2. 构建网络和数据集

该章节构造一个用于对CIFAR10数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[3, 32, 32]`，输出类别数为10。
为了方便展示示例，我们使用Paddle高层API提供的预定义[mobilenetv1分类模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/models/mobilenetv1/MobileNetV1_cn.html#mobilenetv1)。
调用`model.prepare`配置模型所需的部件，比如优化器、损失函数和评价指标，API细节请参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/hapi/model/Model_cn.html#prepare-optimizer-none-loss-function-none-metrics-none)

```python
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

transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = Cifar10(mode='train', backend='cv2', transform=transform)
val_dataset = Cifar10(mode='test', backend='cv2', transform=transform)
```

## 3. 进行预训练

对模型进行预训练，为之后的量化做准备。
执行以下代码对模型进行预训练
```python
model.fit(train_dataset, epochs=5, batch_size=256, verbose=1)
model.evaluate(val_dataset, batch_size=256, verbose=1)
```


## 4. 量化训练

### 4.1 将模型转换为模拟量化模型

当使用普通在线量化时`weight_preprocess_type` 用默认设置None即可，当需要使用PACT在线量化时，则设置为'PACT'。

注意，目前PACT在线量化产出的量化模型，使用PaddleLite在ARM CPU上部署时，精度正确，但是使用PaddleInference在NV GPU和Intel CPU上部署时，可能存在精度问题。所以，请合理选择在线量化方法的种类。

```python
quant_config = {
    # weight preprocess type, default is None and no preprocessing is performed.
    'weight_preprocess_type': None,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}

quanter = QAT(config=quant_config)
quanter.quantize(net)
```

### 4.2 训练量化模型

在这里我们对量化模型进行finetune训练，我们可以看到量化训练得到的模型与原模型准确率十分接近，代码如下所示：

```python
model.fit(train_dataset, epochs=2, batch_size=256, verbose=1)
model.evaluate(val_dataset, batch_size=256, verbose=1)
```

在量化训练得到理想的量化模型之后，我们可以将其导出用于预测部署。


## 5. 导出预测模型

通过以下接口，可以直接导出量化预测模型：

```python
path="./quant_inference_model"
quanter.save_quantized_model(
    net,
    path,
    input_spec=inputs)
```

导出之后，可以在`path`路径下找到导出的量化预测模型。

根据部署业务场景，可以使用PaddleLite将该量化模型部署到移动端（ARM CPU），或者使用PaddleInference将该量化模型部署到服务器端（NV GPU和Intel CPU）。

导出的量化模型相比原始FP32模型，模型体积没有明显差别，这是因为量化预测模型中的权重依旧保存为FP32类型。在部署时，使用PaddleLite opt工具转换量化预测模型后，模型体积才会真实减小。

部署参考文档：
* 部署[文档](../../deploy/index.html)
* PaddleLite部署量化模型[文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/quant_aware.html)
* PaddleInference Intel CPU部署量化模型[文档](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_x86_cpu_int8.html)
* PaddleInference NV GPU部署量化模型[文档](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_trt.html)

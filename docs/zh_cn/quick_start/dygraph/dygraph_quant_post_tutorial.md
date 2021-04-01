#  离线量化

离线量化又称为训练后量化，仅需要使用少量校准数据，确定最佳的量化参数降低量化误差。这种方法需要的数据量较少，但量化模型精度相比在线量化稍逊。

下面该教程将以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的模型量化接口]()。

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

训练完成后导出预测模型:
```python
paddle.jit.save(net, "./fp32_inference_model", input_spec=[inputs])
```


## 4.离线量化

调用slim接口将原模型转换为离线量化模型, 导出的模型可以直接用于预测部署：

```python
paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir='./',
        model_filename='fp32_inference_model.pdmodel',
        params_filename='fp32_inference_model.pdiparams',
        quantize_model_path='./quant_post_static_model',
        sample_generator=train_dataset,
        batch_nums=10)
```

注意，使用离线量化方法后模型精度完全错误，可能是模型中存在控制流OP，目前离线量化方法还不支持对这类模型进行量化。

根据部署业务场景，可以使用PaddleLite将该量化模型部署到移动端（ARM CPU），或者使用PaddleInference将该量化模型部署到服务器端（NV GPU和Intel CPU）。

导出的量化模型相比原始FP32模型，模型体积没有明显差别，这是因为量化预测模型中的权重依旧保存为FP32类型。在部署时，使用PaddleLite opt工具转换量化预测模型后，模型体积才会真实减小。

部署参考文档：
* 部署[文档](../../deploy/index.html)
* PaddleLite部署量化模型[文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/quant_aware.html)
* PaddleInference Intel CPU部署量化模型[文档](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_x86_cpu_int8.html)
* PaddleInference NV GPU部署量化模型[文档](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_trt.html)

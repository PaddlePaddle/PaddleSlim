# LatencyPredictor使用教程

LatencyPredictor主要功能是根据提供的op-latency映射表，预估神经网络在特定硬件设备上的实际耗时。它基于Paddle-Lite开发，适用于使用Paddle-Lite部署的模型。映射表以key-value的形式存储，key包含了神经网络模型经过Paddle-Lite图优化后的各种融合算子（op）信息，value则代表在特定硬件上的实际耗时。同时，还基于这些op的耗时数据，训练了op级别的耗时预测器，用于预测延时映射表中未包含的op耗时数据，实现对任意模型的延时预测。

## 使用方法

1. 构建LatencyPredictor
2. 定义模型，并保存其预测模型
3. 对模型进行延时预估


### 1. 构建LatencyPredictor
设置硬件信息初始化TableLatencyPredictor，将会自动下载对应的延时映射表。如下所示，设置芯片为骁龙710，将在当前目录自动下载延时映射表“SD710_threads_4_power_mode_0.pkl”。该表名说明了op延时数据的采集环境为：线程数threads为4，测速模式power_mode为3，测试batchsize涵盖1、2、4、8。

目前可选硬件类别有骁龙625、710、865('SD625, SD710, SD845')，用户也可输入已有的延时表路径用于延时预估，如predicor2所示。
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')

predictor2 = paddleslim.TableLatencyPredictor(table_file='./table/SD865_threads_4_power_mode_0.pkl')
```

### 2. 定义模型，并保存其预测模型
延时预估器通过读取预测模型文件（*.pdmodel, *.pdiparams），获取模型的拓扑结构和详细的op信息。首先需要保存卷积神经网络的预测模型文件，以下为导出预测模型文件的示例：
```
import paddle
from paddle.vision.models import mobilenet_v1
from paddle.static import InputSpec

model = mobilenet_v1()
x_spec = InputSpec(shape=[1, 3, 224, 224], dtype='float32', name='inputs')
static_model = paddle.jit.to_static(model, input_spec=[x_spec])
paddle.jit.save(static_model, 'mobilenetv1') # 将在当前路径下生成mobilenetv2.pdmodel、mobilenetv2.pdiparams
```
更多关于保存预测模型的api说明，请参考[paddle的api文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html#save)

### 3. 对模型进行延时预估

生成预测模型文件后，可通过predict函数直接预测模型推理延时，同时还可通过set_predictor_state函数开启op预测器功能（默认关闭），能对任意模型进行延时预估（暂时只支持SD625、710）。

需说明的是，该模型的输入大小、数据类型需在保存预测模型时设置好。保存预测模型时，输入大小需要为具体值，暂时不支持可变长格式。若提供的预测模型是fp32类型，则预测该模型fp32格式的推理延时；若预测模型保存为int8类型，则预测该模型int8格式的推理延时。

完整使用示例如下所示：
```
import paddle
from paddle.vision.models import mobilenet_v1
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')

model = mobilenet_v1()
x_spec = InputSpec(shape=[1, 3, 224, 224], dtype='float32', name='inputs')
static_model = paddle.jit.to_static(model, input_spec=[x_spec])
paddle.jit.save(static_model, 'mobilenetv2') # 将在当前路径下生成mobilenetv2.pdmodel、mobilenetv2.pdiparams

# predictor.set_predictor_state(state=True) # 开启预测器功能
latency = predictor.predict(model_file='mobilenetv2.pdmodel', param_file='mobilenetv2.pdiparams)
print('predicted latency = {}ms'.format(latency))
```

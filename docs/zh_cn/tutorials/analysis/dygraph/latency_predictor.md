# LatencyPredictor使用教程

LatencyPredictor主要功能是根据提供的op-latency映射表，预估神经网络网络在特定硬件设备上的实际耗时。它基于Paddle-Lite开发，适用于使用Paddle-Lite部署的模型。映射表以key-value的形式存储，key包含了神经网络模型经过Paddle-Lite图优化后的各种融合op信息，value则代表在特定硬件上的实际耗时。

## 使用方法

1. 准备映射表和opt优化工具
2. 构建LatencyPredictor
3. 定义模型和预测

### 1. 准备映射表和opt优化工具

1.1 映射表可以从[这里](https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/845_threads_4_power_mode_3_batchsize_1.pkl)下载  

1.2 opt编译请参考Paddle-Lite[文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)，要求源码编译。编译时需要关闭Paddle-Lite的内存复用功能，即注释掉这[几行代码](https://github.com/PaddlePaddle/Paddle-Lite/blob/d76f45be989d3e01cebf2ac18e047cfd37d52666/lite/core/optimizer/optimizer.cc#L266-L268)。

### 2. 构建LatencyPredictor

根据提供的映射表(*.pkl)路径和opt工具路径构建LatencyPredictor对象
```python
import paddleslim

table_file = {映射表路径}
opt_path = {opt工具路径}
predictor = paddleslim.TableLatencyPredictor(table_file, opt_path)
```

### 3. 定义模型和预测

定义model后可通过predict_latency函数直接预测模型推理耗时，其中，input_shape为输入大小，save_dir为中间pbmodel模型保存路径，data_type可选fp32或int8，task_type=‘cls'表示该模型为分类模型。
```python
import paddle
from paddle.vision.models import mobilenet_v1

model = mobilenet_v1()
latency = predictor.predict_latency(model, input_shape=[1,3,224,224], save_dir='./model', data_type='int8', task_type='cls')
print('predicted latency = {}ms'.format(latency))
```

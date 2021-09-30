# LatencyPredictor使用教程

LatencyPredictor主要功能是根据提供的op-latency映射表，预估神经网络网络在特定硬件设备上的实际耗时。它基于Paddle-Lite开发，适用于使用Paddle-Lite部署的模型。映射表以key-value的形式存储，key包含了神经网络模型经过Paddle-Lite图优化后的各种融合op信息，value则代表在特定硬件上的实际耗时。

## 使用方法

1. 下载或自行编译opt优化工具
2. 构建LatencyPredictor
3. 定义模型和预测

### 1. 下载或自行编译opt优化工具
1.1 下载提供的opt工具，可根据运行环境下载适用的opt，目前提供Mac平台([M1芯片](https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/opt_M1_mac)，[Intel芯片](https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/opt_intel_mac))和[Ubuntu](https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/opt_ubuntu)平台的opt工具下载。
1.2 也可以自行通过Paddle-Lite源码编译opt工具，具体请参考请参考Paddle-Lite[文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)。编译时需要关闭Paddle-Lite的内存复用功能，即注释掉这[几行代码](https://github.com/PaddlePaddle/Paddle-Lite/blob/d76f45be989d3e01cebf2ac18e047cfd37d52666/lite/core/optimizer/optimizer.cc#L266-L268)。

### 2. 构建LatencyPredictor

提供opt工具路径，以及芯片和测试参数信息，LatencyPredictor会根据这些参数自动下载对应的映射表。如下所示，芯片为845芯片，测试线程数threads为4，测速模式power_mode为3，测试batchsize为1.
```
import paddleslim

opt_path = {opt工具路径}
predictor = paddleslim.TableLatencyPredictor(opt_path, hardware='845', threads=4, power_mode=3, batchsize=1)
```

### 3. 定义模型和预测

定义model后可通过predict_latency函数直接预测模型推理耗时，其中，input_shape为输入大小，save_dir为中间pbmodel模型保存路径，data_type可选fp32或int8，task_type=‘cls'表示该模型为分类模型。
```
import paddle
from paddle.vision.models import mobilenet_v1

model = mobilenet_v1()
latency = predictor.predict_latency(model, input_shape=[1,3,224,224], save_dir='./model', data_type='int8', task_type='cls')
print('predicted latency = {}ms'.format(latency))
```

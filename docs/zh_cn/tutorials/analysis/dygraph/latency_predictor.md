# LatencyPredictor使用教程

延时预估器（LatencyPredictor）用于预估模型在特定硬件设备上的推理延时。在无需部署模型到实际环境的情况下，可以快速预估出多种部署环境和设置下的推理延时。当前，
* 支持所有可以使用 Paddle Lite 部署的模型；
* 支持预估 ARM CPU 上的模型耗时。

## 1. 准备环境
### 1.1 版本要求
```bash
python>=3.7
PaddleSlim>=2.3.0
```
### 1.2 安装 PaddleSlim
* 通过 pip install 的方式进行安装:
```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* 或者从源码安装最新版 PaddleSlim:

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
python3.7 -m pip install -r requirements.txt # 从requirements.txt安装依赖库
```

## 2. 快速开始
### 2.1 准备推理模型
延时预估器通过读取推理模型文件（\*.pdmodel, \*.pdiparams）进行预估。以 MobileNetv1 为例，请从[这里](https://bj.bcebos.com/v1/paddlemodels/PaddleSlim/analysis/mobilenetv1.tar)下载推理模型文件。
```bash
wget https://bj.bcebos.com/v1/paddlemodels/PaddleSlim/analysis/mobilenetv1.tar
tar -xf mobilenetv1.tar
```
使用自定义模型结构时，可参考[api文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html#save)保存推理模型。

### 2.2 预估推理延时
构造 TableLatencyPredictor 类实例，并调用 predict 函数预估推理模型的延时。
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')
latency = predictor.predict(model_file='mobilenetv1_fp32.pdmodel', param_file='mobilenetv1_fp32.pdiparams, data_type='fp32')
print('predicted latency = {}ms'.format(latency))
```
通过设置 table_file 来指定硬件信息，当前支持“SD625”、“SD710”、“SD845”三款骁龙芯片。
> 注1：耗时是基于**保存推理模型时设定的输入形状**预估而得；
>
> 注2：暂时不支持可变长输入，后续将会添加该功能。
## 3. 更多特性
### 3.1 预估模式选择

预估模型延时有两种方式：
* 查表：根据已有的延时表，查找推理模型中每个算子（op）的延时，从而预估模型整体延时。优点是面对表中已覆盖的模型能实现快速准确查找，缺点是面对新模型束手无策；
* 预测器：构建了 op 级别的预测器，作为延时表的补充，能对任意模型进行延时预估。
通过调用 set_predictor_state 函数可开启预测器，选择“查表+预测器”结合的预测方法，如下所示：
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')
predictor.set_predictor_state(True)
```
> op 预测器只预测 batchsize=1 的延时，支持 SD625 和 SD710 设备，默认关闭。后续将在更多设备上扩充不同 batchsize 的 op 预测器。

### 3.2 支持预测 INT8 模型
延时预估器支持对 INT8 量化模型进行延时预估，仅需提供 INT8 量化保存的推理模型文件，并将在调用 predict 函数时，设置 data_type='int8'，如下所示：
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')
predictor.predict(model_file='mobilenetv1_int8.pdmodel', param_file='mobilenetv1_int8.pdiparams, data_type='int8')
```

## 4. 预估效果
延时预估器在 SD625、SD710 等设备上的测速设置都是线程数 threads 为4，测速模式 power_mode 为 0，涵盖了 PaddleClas、PaddleDetection 中的移动端模型，后续将支持其他线程数。下表展示了对典型分类、检测模型在 SD710 的预估效果，预估延时误差均小于 10%。

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<strong>表1: SD710 预测结果</strong>
| Model  | Predict(ms)                       | Real(ms)        | Error(%) |
|:-----:|:----------------------------:|:---------------------:|:--------------------------:|
| MobileNetV1_x0_25|  3.856 | 4.082  |  5.552  |
| MobileNetV1_x0_5|  11.456 | 11.804 |  2.948  |
| MobilenetV1|  39.107 | 39.448  |  0.865   |
| MobileNetV2_x0_5|  9.905 | 10.470  |  5.395   |
| MobilenetV2  | 26.666 | 27.542 | 3.183  |
| MobileNetV2_x2_0  | 86.281 | 86.824 | 0.625  |
| MobileNetV3_large_x0_35 | 6.428 | 6.911 | 6.984    |
| MobileNetV3_large_x1_0 | 21.566 | 23.108 | 6.673    |
| MobileNetV3_large_x1_25 | 32.888 | 33.641 | 2.236    |
| GhostNet_x0_5  | 8.294 | 9.182  |  9.675   |
| GhostNet_x1_0  | 18.603 | 19.916 |  6.594   |
| GhostNet_x1_3   | 26.896 | 28.0525  |  4.120  |
| ShuffleNetV2_x1_0  | 13.199   |  14.476  |  8.825   |
| ShuffleNetV2_x1_5  | 23.066   |  25.082  |  8.038   |
| ShuffleNetV2_x2_0  | 41.379   |  43.868  |  5.674   |
| ppyolo_mbv3_large_coco  | 70.055  |  72.063  |  2.787   |
| ppyolo_tiny_650e_coco  | 43.808   |  45.3393  |  3.377   |
| picodet_l_320_coco  | 92.603   |  92.926  |  0.347   |
| picodet_m_320_coco  | 69.176   |  65.778  |  4.911   |
| picodet_s_320_coco  | 38.874   |  36.999  |  4.823   |

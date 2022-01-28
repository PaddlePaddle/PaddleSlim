# LatencyPredictor使用教程

LatencyPredictor主要功能是根据提供的op-latency映射表和已训练的op预测器，预估神经网络在特定硬件设备上的实际耗时。它基于Paddle-Lite开发，适用于使用Paddle-Lite部署的模型。

## 1. 准备环境

安装 PaddleSlim>=2.3.0。由于LatencyPredictor基于Paddle-Lite开发，python需至少为3.7版本。

* 可以通过 pip install 的方式进行安装。

```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* 如果获取 PaddleSlim 的最新特性，可以从源码安装。

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
python3.7 -m pip install -r requirements.txt # 从requirements.txt安装依赖库
```

## 2. 快速开始
### 2.1 准备预测模型
延时预估器通过读取预测模型文件（*.pdmodel, *.pdiparams）进行预估。以mobilenetv1为例，请从[这里](https://bj.bcebos.com/v1/paddlemodels/PaddleSlim/analysis/mobilenetv1.tar)下载其预测模型文件。使用自定义模型结构时，可参考[api文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html#save)，保存预测模型。
```bash
wget https://bj.bcebos.com/v1/paddlemodels/PaddleSlim/analysis/mobilenetv1.tar
tar -xf mobilenetv1.tar
```
### 2.2 预测
设置硬件信息初始化TableLatencyPredictor，然后调用predict函数进行预测。目前可选硬件类别有骁龙625、710、865 ('SD625, SD710, SD845')。
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')
latency = predictor.predict(model_file='mobilenetv1_fp32.pdmodel', param_file='mobilenetv1_fp32.pdiparams, data_type=fp32)
print('predicted latency = {}ms'.format(latency))
```
> 注1：预估的耗时是基于**保存预测模型时设定的输入形状**预估而得；
>
> 注2：暂时不支持可变长输入，后续将会添加该功能。
## 3. 更多特性
### 3.1 op预测器
我们基于这些op的耗时数据，训练了op级别的耗时预测器，用于预测延时映射表中未包含的op耗时数据，实现对任意模型的延时预测。目前，实现了在SD625和SD710上的op预测器，可通过set_predictor_state函数开启op预测器功能，如下所示。
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')
predictor.set_predictor_state(True)
```
> 该功能默认关闭，此时仅依赖延时表预测耗时。op预测器只预测batchsize=1的延时，后续将在更多设备上扩充不同batchsize的op预测器。

### 3.2 支持预测int8模型
我们的延时预估器还支持对int8量化模型进行延时预估，仅需提供int8量化保存的预测模型文件，并将设置predict函数data_type=int8即可，如下所示。
```
import paddleslim

predictor = paddleslim.TableLatencyPredictor(table_file='SD710')
predictor.predict(model_file='mobilenetv1_int8.pdmodel', param_file='mobilenetv1_int8.pdiparams, data_type=int8)
```

## 4. 预测效果
目前，我们在骁龙625、710等设备上的测速设置都是线程数threads为4，测速模式power_mode为0，涵盖了PaddleClas、PaddleDetection中的移动端模型，后续将扩展其他线程数（threads=1，2）的延时表。下表展示了在骁龙710上预测效果，在典型分类、检测模型上都达到了预测误差小于10%。

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<strong>表1: 骁龙710上预测结果</strong>
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

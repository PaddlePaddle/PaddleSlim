# Low-precision Training & Quantization-aware Training With PaddlePaddle

## Introduction
**Quantization-aware Training:** network quantization for inference acceleration, i.e., the weights and activations are quantized into low-bit integers so that the forward pass can be converted into integer matrix multiplications.

**Low-precision Training:** network quantization for training acceleration, i.e., the weights, activations as well as backpropagated errors are quantized into low-bit integers so that the forward and backward pass can be converted into integer matrix multiplications.


## Environment Configuration
python 3.10 \
paddlepaddle-gpu 2.6
```sh
cd classification/
pip install -r requirements.txt
cd quantization
python setup.py build
install_paddle_so.py build/cudnn_convolution_custom/lib.linux-x86_64-cpython-310/cudnn_convolution_custom.so
```

## Train
Low-precision Training: 
```sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m paddle.distributed.launch --gpus="4,5,6,7" train_qt.py -c ../ppcls/configs/ImageNet/ResNet/ResNet18_custom.yaml
```

```py
config.qconfigdir = '../qt_config/qt_W4_mse_det__A4_mse_det__G4_minmax_sto.yaml'
```
Quantization-Aware Training
```sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m paddle.distributed.launch --gpus="4,5,6,7" train_qat.py -c ../ppcls/configs/ImageNet/ResNet/ResNet18_custom.yaml
```

```py
config.qconfigdir = '../qat_config/qat_w4_mse_a4_mse.yaml'
```
## Evalute
```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" eval_qt.py -c ../ppcls/configs/ImageNet/ResNet/ResNet18_custom.yaml
```

## Results
Datastes: Imagenet ilsvrc12 \
Model: Resnet18
|  config   | top-1(%)  | top-5(%)  |
|  ----  | ----  | ----  |
| original train  | 70.852 | 89.700 |
| low-precision train (test with w4a4)  | 66.498 | 86.858 |
| low-precision train (test with w4)  | 67.200 | 87.196 |
| low-precision train (test with fp)    | 66.330 | 86.790 |

## 目录结构
```
/classification
├── README.md           # 项目说明文件
├── requirements.txt    # 项目依赖
├── tool/               # 源代码目录
│   ├── train_py.py     # 量化训练主程序入口
│   ├── train.py        # 全精度训练主程序入口
│   ├── eval_qt.py      # 测试代码入口
├── ppcl/               # 分类网络功能实现模块
├── qt_config/          # 训练量化配置脚本，在此处更改量化配置
├── qat_config/         # Quantization-Aware Training config
└── quantization/       # 量化训练功能实现模块
```

## 量化参数具体说明
以`qt_W4_minmax_det__A4_mse_det__G4_minmax_sto.yaml`为例：

`quantization_manager`: 'qt_quantization_manager'为 Low-precision Training，'qat_quantization_manager'为 Quantization-Aware Training

#### weight_forward_config 正向权重量化参数
`qtype`: "int4"
量化类型。表示权重使用 4 位整数（int4）进行量化。

`granularity`: "N"
量化粒度。"N" 表示按神经网络的各个层级进行量化。

`scaling`: "dynamic"
缩放方法。"dynamic" 表示动态缩放，量化过程中根据数据动态调整缩放因子。

`observer`: "minmax"
量化损失类型。"minmax" 表示使用最小最大值来优化量化参数。

`stochastic`: False
是否使用随机量化。False 表示不使用随机量化。

`filters`:
用于具体指定哪些层或部分进行量化。

id: 0, qtype: "full": 表示 ID 为 0 的层（通常是第一个层）使用全量化（"full"）。
name: "fc", qtype: "full": 表示名称为 fc 的层（通常是全连接层）使用全量化。

#### weight_backward_config 反向传播权重量化参数
`granularity`: "C"
量化粒度。"C" 表示按通道（channel）进行量化。

#### input_config 输入量化参数
`observer`: "mse"
量化损失类型。"mse" 表示使用均方误差（Mean Squared Error）损失优化量化参数。

#### grad_input_config 输入梯度量化参数
granularity: "NHW"
量化粒度。"NHW" 表示按输入数据的每个维度（batch size, height, width）进行量化。

#### grad_weight_config 权重梯度量化参数
granularity: "NC"
量化粒度。"NC" 表示按通道和输出维度进行量化。


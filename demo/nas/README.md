# SANAS网络结构搜索示例

本示例介绍如何使用网络结构搜索接口，搜索到一个更小或者精度更高的模型，该示例介绍paddleslim中SANAS的使用及如何利用SANAS得到模型结构，完整示例代码请参考sa_nas_mobilenetv2.py或者block_sa_nas_mobilenetv2.py。

## 数据准备
本示例默认使用cifar10数据，cifar10数据会根据调用的paddle接口自动下载，无需额外准备。

## 接口介绍
请参考<a href='../../docs/zh_cn/api_cn/nas_api.rst'>神经网络搜索API文档</a>。

本示例为利用SANAS在MobileNetV2的搜索空间上搜索FLOPs更小的模型。
## 1 搜索空间配置
默认搜索空间为`MobileNetV2`，详细的搜索空间配置请参考<a href='../../docs/zh_cn/api_cn/search_space.md'>搜索空间配置文档</a>。

## 2 启动训练

### 2.1 启动基于MobileNetV2初始模型结构构造搜索空间的实验
```shell
CUDA_VISIBLE_DEVICES=0 python sa_nas_mobilenetv2.py
```


### 2.2 启动基于MobileNetV2的block构造搜索空间的实验
```shell
CUDA_VISIBLE_DEVICES=0 python block_sa_nas_mobilenetv2.py
```

# RLNAS网络结构搜索示例

本示例介绍如何使用RLNAS接口进行网络结构搜索，该示例介绍paddleslim中RLNAS的使用，完整示例代码请参考rl_nas_mobilenetv2.py或者parl_nas_mobilenetv2.py。

## 数据准备
本示例默认使用cifar10数据，cifar10数据会根据调用的paddle接口自动下载，无需额外准备。

## 接口介绍
请参考<a href='../../docs/zh_cn/api_cn/nas_api.rst'>神经网络搜索API文档</a>。

示例为利用RLNAS在MobileNetV2的搜索空间上搜索精度更高的模型。
## 1 搜索空间配置
默认搜索空间为`MobileNetV2`，详细的搜索空间配置请参考<a href='../../docs/zh_cn/api_cn/search_space.md'>搜索空间配置文档</a>。

## 2 启动训练

### 2.1 启动基于MobileNetV2初始模型结构构造搜索空间，强化学习算法为lstm的搜索实验
```shell
CUDA_VISIBLE_DEVICES=0 python rl_nas_mobilenetv2.py
```

### 2.2 启动基于MobileNetV2初始模型结构构造搜索空间，强化学习算法为ddpg的搜索实验
```shell
CUDA_VISIBLE_DEVICES=0 python parl_nas_mobilenetv2.py
```

# 自然语言处理模型定点量化教程（采用训练中引入量化策略）

（本量化教程使用MKL-DNN，针对量化NLP模型Ernie。其他NLP模型后续会根据需求继续优化）

## 概述

量化是模型压缩的重要手段，可以缩小模型，并且保证精度仅有极小的下降。在PaddlePaddle中，量化策略`post`为使用离线量化得到的模型，`aware`为在线量化训练得到的模型。本教程介绍了使用训练时量化策略(`aware`)，结合MKL-DNN库，对NLP模型Ernie进行量化和预测加速。经过量化和MKL-DNN预测加速后，INT8模型在单线程上性能为原FP32模型的2.68倍，而精度仅下降了0.24%。此次Ernie模型量化实现了以下优势（特性）。

在训练脚本中，在主要算子前插入量化op和反量化op，和因为op融合需要的quantize_dequantize_pass，通过训练微调这两类op。目前我们支持以下op前插入量化和反量化op：
```
conv, depthwise_conv2d, mul (anything else)
```
在转化成真实定点模型（INT8模型）阶段，根据MKL-DNN支持，将激活函数合入到主op中。我们目前可以实现以下pattern的INT8 fuse如下。 op融合后不仅使用INT8 计算，而且融合了激活函数，无需另外开辟空间，大大提高了性能。
```
conv+elementwise_add -> fc
matmul+reshape+transpose -> matmul
```

注意：
1. 需要MKL-DNN和MKL。 只有使用AVX512系列CPU服务器才能获得性能提升。
2. 在支持AVX512 VNNI扩展的CPU服务器上，INT8精度最高。

## 1. 安装PaddleSlim

可按照[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)中的步骤安装PaddleSlim。

PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 训练

根据 [tools/train.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/tools/train.py) 编写压缩脚本train.py。脚本中量化的步骤如下。

### 2.1 预训练或者下载预训练好的模型
* 用户可以在此处链接下载我们已经预训练好的的模型。[预训练模型下载](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)

### 2.2 插入伪造量化反量化OP
在Program中插入量化和反量化OP。`paddleslim.quant.quant_aware` 作用是在网络中的conv2d、depthwise_conv2d、mul等算子的各个输入前插入连续的量化op和反量化op，并改变相应反向算子的某些输入。并且在一些op之后加上quant_dequant op, 示例图如下：
<p align="center">
<img src="./images/TransformPass.png" height=400 width=520 hspace='10'/> <br />
<strong>图1：应用 paddleslim.quant.quant_aware 后的结果</strong>
</p>
对应到代码中的更改，首先需要更改 `config` 配置如下。Ernie模型所需要全部`config`设置已经列出。如果想了解各参数含义，可参考 [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)

```
config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d','elmentwise_add','pool2d']
    }
```

然后在`train.py`中调用quant_aware

```
quant_program  = quant_aware(train_prog, place, config, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)
```

### 2.2 关闭一些训练策略

因为量化要对Program做修改，所以一些会修改Program的训练策略需要关闭。``sync_batch_norm`` 和量化多卡训练同时使用时会出错，原因暂不知，因此也需要将其关闭。
```
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
```

### 2.3 边训练边测试量化后的模型

* 用户可以使用train.py进行训练
调用train函数训练分类网络，train_program是在第2步：构建网络中定义的。
`train(train_program)`
调用test函数测试分类网络，val_program是在第2步：构建网络中定义的。
`test(val_program)`

### 2.1-2.3 训练示例

您可以通过运行以下命令运行该示例。

step1: 设置gpu卡
```
export CUDA_VISIBLE_DEVICES=0
```
step2: 开始训练

请在PaddleDetection根目录下运行。

```
python slim/quantization/train.py --not_quant_pattern yolo_output \
    --eval \
    -c ./configs/yolov3_mobilenet_v1.yml \
    -o max_iters=30000 \
    save_dir=./output/mobilenetv1 \
    LearningRate.base_lr=0.0001 \
    LearningRate.schedulers="[!PiecewiseDecay {gamma: 0.1, milestones: [10000]}]" \
    pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
```

>通过命令行覆设置max_iters选项，因为量化的训练轮次比正常训练小很多，所以需要修改此选项。
如果要调整训练卡数，可根据需要调整配置文件`yolov3_mobilenet_v1_voc.yml`中的以下参数：

- **max_iters:** 训练的总轮次。
- **LeaningRate.base_lr:** 根据多卡的总`batch_size`调整`base_lr`，两者大小正相关，可以简单的按比例进行调整。
- **LearningRate.schedulers.PiecewiseDecay.milestones：** 请根据batch size的变化对其调整。


通过`python slim/quantization/train.py --help`查看可配置参数。
通过`python ./tools/configure.py help ${option_name}`查看如何通过命令行覆盖配置文件中的参数。


### 2.4 保存断点（checkpoint）
在脚本中使用保存checkpoint的代码为：
```
# insert quantize op in eval_prog
eval_prog = quant_aware(eval_prog, place, config, for_test=True)
checkpoint.save(exe, eval_prog, os.path.join(save_dir, save_name))
```

## 3. 转化

### 转化模型为定点模型

``paddleslim.quant.convert`` 主要用于改变Program中量化op和反量化op的顺序，即将类似图1中的量化op和反量化op顺序改变为图2中的布局。除此之外，``paddleslim.quant.convert`` 还会将`conv2d`、`depthwise_conv2d`、`mul`等算子参数变为量化后的int8_t范围内的值(但数据类型仍为float32)，示例如图2：

<p align="center">
<img src="./images/FreezePass.png" height=400 width=420 hspace='10'/> <br />
<strong>图2：paddleslim.quant.convert 后的结果</strong>
</p>

所以在调用 ``paddleslim.quant.convert`` 之后，才得到最终的量化模型。此模型可使用PaddleLite进行加载预测，可参见教程[Paddle-Lite如何加载运行量化模型](https://github.com/PaddlePaddle/Paddle-Lite/wiki/model_quantization)。

### 4. 评估脚本
使用脚本[slim/quantization/eval.py](./eval.py)进行评估。

- 定义配置。使用和训练脚本中一样的量化配置，以得到和量化训练时同样的模型。
- 使用 ``paddleslim.quant.quant_aware`` 插入量化和反量化op。
- 使用 ``paddleslim.quant.convert`` 改变op顺序，得到最终量化模型进行评估。

评估命令：

```
python slim/quantization/eval.py --not_quant_pattern yolo_output  -c ./configs/yolov3_mobilenet_v1.yml \
-o weights=./output/mobilenetv1/yolov3_mobilenet_v1/best_model
```

## 5. 导出模型

使用脚本[slim/quantization/export_model.py](./export_model.py)导出模型。

- 定义配置。使用和训练脚本中一样的量化配置，以得到和量化训练时同样的模型。
- 使用 ``paddleslim.quant.quant_aware`` 插入量化和反量化op。
- 使用 ``paddleslim.quant.convert`` 改变op顺序，得到最终量化模型进行评估。

导出模型命令：

```
 python slim/quantization/export_model.py --not_quant_pattern yolo_output  -c ./configs/yolov3_mobilenet_v1.yml --output_dir ${save path} \
-o weights=./output/mobilenetv1/yolov3_mobilenet_v1/best_model
```

## 6. MKL-DNN 模型融合加速

## 7. 预测

### python预测

在脚本<a href="./infer.py">slim/quantization/infer.py</a>中展示了如何使用fluid python API加载使用预测模型进行预测。

运行命令示例:
```
python slim/quantization/infer.py --not_quant_pattern yolo_output \
-c ./configs/yolov3_mobilenet_v1.yml \
--infer_dir ./demo \
-o weights=./output/mobilenetv1/yolov3_mobilenet_v1/best_model
```


### PaddleLite预测
导出模型步骤中导出的FP32模型可使用PaddleLite进行加载预测，可参见教程[Paddle-Lite如何加载运行量化模型](https://github.com/PaddlePaddle/Paddle-Lite/wiki/model_quantization)


## 复现结果

### Ernie 精度和性能 on [XNLI dataset](https://github.com/facebookresearch/XNLI)

>**I. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 的精度结果**

|     Model    |  FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |          80.20%        |         79.96%   |     -0.24%      |  


>**II. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 上单样本耗时**

|     Threads  | FP32 Latency (ms) | QAT INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:-----------------:|
| 1 thread     |        252.13          |            93.80    |     2.69x         |
| 20 threads   |        29.19           |            17.38    |     1.68x         |



## FAQ

该示例使用PaddleSlim提供的[量化压缩API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)对检测模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [检测模型的常规训练方法](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)

已发布量化模型见[压缩模型库](../README.md)

# 图像分类模型定点量化教程

## 概述

量化是模型压缩的重要手段。在PaddlePaddle中，量化策略`post`为使用离线量化得到的模型，`aware`为在线量化训练得到的模型。使用slim的量化训练`quant-aware` 和离线量化 `post training`产出量化模型，都使用相同的方法在cpu上进行预测部署。在Intel(R) Xeon(R) Gold 6271机器上，量化后的INT8模型在单线程上性能为原FP32模型的3~4倍，而精度仅有极小下降。目前，我们目前支持量化op包括conv2d、depthwise_conv2d、mul、matmul、pool2d、transpose、reshape等；同时在DNNL优化阶段，我们会fuse很多其他op，包括batch_norm、relu、brelu，elementwise_add等，经过Op量化和op fuses，量化模型性能会大大提升。具体DNNL优化可以参考[SLIM QAT for INT8 DNNL](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/QAT_mkldnn_int8_readme.md)。本文以图像分类模型为样例描述整个量化过程，NLP模型量化过程类似，具体参考[ERNIE 模型 QAT INT8 精度与性能复现](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn)

## 1. 准备

#### 1.0 预测部署前提
- 只有使用AVX512系列CPU服务器才能获得性能提升。用户可以通过在命令行红输入`lscpu`查看本机支持指令。
- 在支持`avx512_vnni`的CPU服务器上，INT8精度最高，性能提升最快。

#### 1.1 准备预测推理库

用户可以从源码编译Paddle推理库，也可以直接下载推理库。
- 从Paddle源码编译Paddle推理库，请参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12)。如果用户从源码编译，使用release/1.8版本，下面两个选项在此次编译中请打开

| 选项        | 值           | 说明  |
| ------------- |:-------------:| -----:|
| WITH_MKL      | ON | 本测试请打开MKL |
| WITH_MKLDNN   | ON | 本测试请打开MKLDNN |

- 从Paddle官网下载发布的[预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。请选择`ubuntu14.04_cpu_avx_mkl` 最新发布版或者develop版。

你可以将准备好的预测库重命名为fluid_inference，放置在该测试项目下面，也可以在cmake时通过设置PADDLE_ROOT来指定Paddle预测库的位置。

#### 1.2 从源代码构建PaddleSlim
PaddleSlim 安装请参考[官方安装文档](https://paddlepaddle.github.io/PaddleSlim/install.html)安装develop版本
```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```
#### 1.3 在代码中使用
在用户自己的测试样例中，按以下方式导入Paddle和PaddleSlim:
```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 训练并保存QAT FLOAT32模型
首先，用户在此处链接下载我们已经预训练好的模型：[预训练模型下载](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)。

使用量化训练方法产出量化模型，并且保存QAT FLOAT32模型的脚本。量化训练流程可以参考 [分类模型的离线量化流程](https://paddlepaddle.github.io/PaddleSlim/tutorials/quant_aware_demo/)和 量化API(https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)。在本示例中，你可以直接运行：
```
python train_image_classification.py --model=ResNet50 --pretrained_model=$PATH_TO_ResNet50_pretrained --data=imagenet --data_dir=/PATH/TO/ILSVRC2012/ --save_float32_qat_dir=/PATH/TO/SAVE/FLOAT32/QAT/MODEL
```
**训练阶段参数说明：**
- **model：** 模型名称，默认值`ResNet50`
- **pretrained_model:** 加载预训练模型路径，默认值为空
- **batch_size：** 训练batch大小，默认值128
- **num_epochs:** 训练回合数，默认值: 1
- **config_file:** 配置文件位置，默认值`./config.yaml`.

如果用户需要更改量化策略，可以更改 `config.yaml` 配置。目前我们建议使用以下配置。
```
config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d','matmul']
    }
```
**`config.yaml` 参数说明：**
- **quantize_op_types:** 目前支持 `depthwise_conv2d`, `mul`, `conv2d`, `matmul`, `transpose2`, `reshape2`, `pool2d`, `scale`的量化。但是训练阶段插入fake quantize/dequantize op时，只需在前四种op前后插入fake quantize/dequantize ops，后面四种op `matmul`, `transpose2`, `reshape2`, `pool2d`, 由于输入输出scale不变，将从前后方op的输入输出scales获得scales,所以`quantize_op_types` 参数只需要 `depthwise_conv2d`, `mul`, `conv2d`, `matmul` 即可
- **其他参数:** 请参考 [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)

## 3. 转化FP32 QAT模型为DNNL优化后的INT8模型
上一步中训练后保存的模型是float32 qat模型。我们还需要移除fake quantize/dequantize op，fuse一些op，并且完全转化成 INT8 模型。需要使用Paddle所在目录运行下面的脚本，脚本在官网的位置为[save_qat_model.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_qat_model.py)。复制脚本到本案例所在目录下并执行如下命令：
```
python save_qat_model.py --qat_model_path=/PATH/TO/SAVE/FLOAT32/QAT/MODEL --int8_model_save_path=/PATH/TO/SAVE/INT8/MODEL --fp32_model_save_path=/PATH/TO/SAVE/FLOAT32/MODEL --quantized_ops="conv2d,pool2d"
```
**参数说明：**
- **qat_model_path:** 为FP32 QAT-> INT8 转化过程的输入模型路径。请设置为上一步在线训练保存的QAT FLOAT32模型所在路径。
- **fp32_model_save_path:** 可选。如果设置，则将QAT FP32模型经过DNNL优化后保存成FP32模型路径。
- **int8_model_save_path:** 可选。如果设置，则为QAT模型转化并经过DNNL优化量化后的INT8模型路径。
- **quantized_ops:** 最终INT8模型中使用量化op的列表，用户设置时需要注意：
  - 只能选择目前已经支持DNNL量化的op可以在列表中。目前支持的DNNL量化op列表是`conv2d`, `depthwise_conv2d`, `mul`, `fc`, `matmul`, `pool2d`, `reshape2`, `transpose2`, `concat`。
  - 所有列表中的即将使用量化op的op的scales不能为空. 否则，量化过程会报错。
  - 有时，量化所有可量化的Op不一定性能最优，这也是为什么我们提供参数给用户选择量化op的原因。比如，如果一个op是单个的INT8 op,不可以与之前的和之后的op融合，那么为了量化这个op，需要先做quantize,然后运行这个INT8 op,再dequantize, 这样的过程可能导致最终性能不如保持该op为fp32 op。
  - 一个找到最优配置的方法是，用户观察这个模型一共用到了哪些可量化的op，然后选出不同的`quantized_ops`组合，多运行几次。 比如，对于图像分类模型来说，我们发现对于图像分类任务， `--quantized_ops=“conv2d, pool2d"`量化后性能最优。对于NLP任务，如Ernie模型， `--quantized_ops="fc,reshape2,transpose2,matmul"`量化后最优。

这个脚本的`qat_model_path`为输入模型，来自上一步中的在线训练量化保存的模型。`fp32_model_save_path`和`int8_model_save_path`为输出模型。如果用户设置了`fp32_model_save_path`路径，则会产出QAT FLOAT32 模型经过DNNL优化后的模型（包括移除fake ops, 使用DNNL kernel和fuses等），调用的优化过程如下，预测速度会比原QAT模型快。如果用户设置了`int8_model_save_path`,则会经过收集scales，反量化成FP32模型，再进行fuse等，再进行DNNL量化。

* save_qat_model 将QAT转化FP32模型, 使用`Qat2Int8MkldnnPass` apply_fp32
```
    def apply_fp32(self, graph):
        graph = self._gather_weight_scales_from_fake(graph)
        graph = self._gather_output_scales_from_attr(graph)
        graph = self._gather_input_scales_from_fake(graph)
        graph = self._remove_fake_ops(graph)
        graph = self._dequantize_weights(graph)
        graph = self._optimize_fp32_graph(graph)
        graph = self._remove_unused_var_nodes(graph)
        graph = self._cleanup(graph)
        return graph
```
* save_qat_model 将QAT转化INT8模型,使用`Qat2Int8MkldnnPass` apply
```
    def apply(self, graph):
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'

        graph = self._gather_weight_scales_from_fake(graph)
        graph = self._gather_output_scales_from_attr(graph)
        graph = self._gather_input_scales_from_fake(graph)
        graph = self._remove_fake_ops(graph)
        graph = self._dequantize_weights(graph)
        graph = self._optimize_fp32_graph(graph)
        graph = self._compute_weight_scales(graph)
        graph = self._update_relu_output_scales(graph)
        graph = self._propagate_scales(graph)
        graph = self._set_dummy_out_scales(graph)
        graph = self._quantize_fp32_graph(graph)
        graph = self._optimize_int8_graph(graph)
        graph = self._cleanup(graph)
        return graph
```
## 4. 预测

### 4.1 数据预处理转化
在精度和性能预测中，需要先对数据进行二进制转化。运行脚本如下可转化完整ILSVRC2012 val数据集。使用`--local`可以转化用户自己的数据。在Paddle所在目录运行下面的脚本。脚本在官网位置为[full_ILSVRC2012_val_preprocess.py](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py)
```
python Paddle/paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py --local --data_dir=/PATH/TO/USER/DATASET/  --output_file=/PATH/TO/SAVE/BINARY/FILE
```

可选参数：
- 不设置任何参数。脚本将下载 ILSVRC2012_img_val数据集，并转化为二进制文件。
- **local:** 设置便为true，表示用户将提供自己的数据
- **data_dir:** 用户自己的数据目录
- **label_list:** 图片路径-图片类别列表文件，类似于`val_list.txt`
- **output_file:** 生成的binary文件路径。
- **data_dim:** 预处理图片的长和宽。默认值 224。

用户自己的数据集目录结构应该如下
```
imagenet_user
├── val
│   ├── ILSVRC2012_val_00000001.jpg
│   ├── ILSVRC2012_val_00000002.jpg
|   |── ...
└── val_list.txt
```
其中，val_list.txt 内容应该如下：
```
val/ILSVRC2012_val_00000001.jpg 0
val/ILSVRC2012_val_00000002.jpg 0
```

注意：
- 为什么将数据集转化为二进制文件？因为paddle中的数据预处理（resize, crop等）都使用pythong.Image模块进行，训练出的模型也是基于Python预处理的图片，但是我们发现Python测试性能开销很大，导致预测性能下降。为了获得良好性能，在量化模型预测阶段，我们决定使用C++测试，而C++只支持Open-CV等库，Paddle不建议使用外部库，因此我们使用Python将图片预处理然后放入二进制文件，再在C++测试中读出。用户根据自己的需要，可以更改C++测试以使用open-cv库直接读数据并预处理，精度不会有太大下降。我们还提供了python测试`sample_tester.py`作为参考，与C++测试`sample_tester.cc`相比，用户可以看到Python测试更大的性能开销。

### 4.2 编译运行预测
#### 编译应用
在样例所在目录下，执
```
mkdir build
cd build
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=$PADDLE_ROOT ..
make -j
```

#### 运行测试
```
# Bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
# Turbo Boost was set to OFF using the command
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# In the file run.sh, set `MODEL_DIR` to `/PATH/TO/SAVE/FLOAT32/QAT/MODEL`或者`/PATH/TO/SAVE/INT8/MODEL`
# In the file run.sh, set `DATA_FILE` to `/PATH/TO/SAVE/BINARY/FILE`
# For 1 thread performance:
./run.sh
# For 20 thread performance:
./run.sh -1 20
```

运行时需要配置以下参数：
- **infer_model:** 模型所在目录，注意模型参数当前必须是分开保存成多个文件的。可以设置为`PATH/TO/SAVE/INT8/MODEL`, `PATH/TO/SAVE/FLOAT32/MODEL`或者`PATH/TO/SAVE/FLOAT32/QAT/MODEL`。无默认值。
- **infer_data:** 测试数据文件所在路径。注意需要是经`full_ILSVRC2012_val_preprocess`转化后的binary文件。
- **batch_size:** 预测batch size大小。默认值为50。
- **iterations:** 预测多少batches。默认为0，表示预测infer_data中所有batches (image numbers/batch size)
- **num_threads:** 预测使用CPU 线程数，默认为单核一个线程。
- **with_accuracy_layer:** 由于这个测试是Image Classification通用的测试，既可以测试float32模型也可以INT8模型，模型可以包含或者不包含label层，设置此参数更改。
- **use_profile:** 由Paddle预测库中提供，设置用来进行性能分析。默认值为false。

你可以直接修改`run.sh`中的MODEL_DIR和DATA_DIR后，即可执行`./run.sh`进行CPU预测。

## 5. QAT量化图像分类模型在 Xeon(R) 6271 和 Xeon(R) 6148 上的精度和性能

表格中的性能是在以下前提获得：
* 通过设置将thread指定给core

   ```
   export KMP_AFFINITY=granularity=fine,compact,1,0
   export KMP_BLOCKTIME=1
   ```

* 使用以下命令将Turbo Boost设置为OFF

   ```
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

### 5.1 QAT量化图片分类任务精度和性能

>**I. QAT DNNL 在 Intel(R) Xeon(R) Gold 6271的精度**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

>**II. QAT DNNL 在 Intel(R) Xeon(R) Gold 6148 的精度**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.85%         |   0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.08%         |   0.18%   |       90.56%       |         90.66%         |  +0.10%   |
|  ResNet101   |       77.50%       |         77.51%         |   0.01%   |       93.58%       |         93.50%         |  -0.08%   |
|   ResNet50   |       76.63%       |         76.55%         |  -0.08%   |       93.10%       |         92.96%         |  -0.14%   |
|    VGG16     |       72.08%       |         71.72%         |  -0.36%   |       90.63%       |         89.75%         |  -0.88%   |
|    VGG19     |       72.57%       |         72.08%         |  -0.49%   |       90.84%       |         90.11%         |  -0.73%   |

>**III. QAT DNNL 在 Intel(R) Xeon(R) Gold 6271的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      73.98      |       227.73        |       3.08        |
| MobileNet-V2 |      86.59      |       206.74        |       2.39        |
|  ResNet101   |      7.15       |        26.69        |       3.73        |
|   ResNet50   |      13.15      |        49.33        |       3.75        |
|    VGG16     |      3.34       |        10.15        |       3.04        |
|    VGG19     |      2.83       |        8.67         |       3.07        |


>**IV. QAT DNNL 在 Intel(R) Xeon(R) Gold 6148的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      75.23      |       111.15        |       1.48        |
| MobileNet-V2 |      86.65      |       127.21        |       1.47        |
|  ResNet101   |      6.61       |        10.60        |       1.60        |
|   ResNet50   |      12.42      |        19.74        |       1.59        |
|    VGG16     |      3.31       |        4.74         |       1.43        |
|    VGG19     |      2.68       |        3.91         |       1.46        |


### 5.2 QAT量化NLP任务精度和性能
>**I. Intel(R) Xeon(R) Gold 6271 精度**

|     Model    |  FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |          80.20%        |         79.64%   |     -0.56%      |  

>**II. Intel(R) Xeon(R) Gold 6148 精度**

| Model | FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
| :---: | :-----------: | :---------------: | :-----------: |
| Ernie |    80.20%     |      79.64%       |    -0.56%     |

>**III. Ernie Intel(R) Xeon(R) Gold 6271 上单样本耗时**

|     Threads  | FP32 Latency (ms) | QAT INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:-----------------:|
| 1 thread     |        236.72          |             83.20    |      2.85x       |
| 20 threads   |        27.40           |            14.99     |      1.83x       |

>**IV. Ernie Intel(R) Xeon(R) Gold 6148 上单样本耗时**

| Model |  Threads   | FP32 Latency (ms) | QAT INT8 Latency (ms) | Ratio (FP32/INT8) |
| :---: | :--------: | :---------------: | :-------------------: | :---------------: |
| Ernie |  1 thread  |    248.42         |       169.30           |       1.46       |
| Ernie | 20 threads |    28.92          |       20.83            |       1.39       |

## FAQ

该示例使用PaddleSlim提供的[量化压缩API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)对检测模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

# Intel CPU量化部署

## 概述

本文主要介绍在CPU上转化PaddleSlim产出的量化模型并部署和预测的流程。对于常见图像分类模型，在Casecade Lake机器上（例如Intel® Xeon® Gold 6271、6248，X2XX等），INT8模型进行推理的速度通常是FP32模型的3-3.7倍；在SkyLake机器（例如Intel® Xeon® Gold 6148、8180，X1XX等）上，使用INT8模型进行推理的速度通常是FP32模型的1.5倍。

流程步骤如下：
- 产出量化模型：使用PaddleSlim训练并产出量化模型。注意模型中被量化的算子的参数值应该在INT8范围内，但是类型仍为float型。
- 在CPU上转换量化模型：在CPU上使用DNNL库转化量化模型为INT8模型。
- 在CPU上部署预测：在CPU上部署样例并进行预测。

参考资料：
* PaddleInference Intel CPU部署量化模型[文档](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_x86_cpu_int8.html)

## 1. 准备

#### 安装Paddle和PaddleSlim

Paddle和PaddleSlim版本必须配套安装。

Paddle 安装请参考[官方安装文档](https://www.paddlepaddle.org.cn/install/quick)。

PaddleSlim 安装请参考[官方安装文档](https://github.com/PaddlePaddle/PaddleSlim)。

#### 在代码中使用

在用户自己的测试样例中，按以下方式导入Paddle和PaddleSlim:
```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 用PaddleSlim产出量化模型

用户可以使用PaddleSlim产出量化训练模型或者离线量化模型。如果用户只想要验证部署和预测流程，可以跳过 2.1 和 2.2, 直接下载[mobilenetv2 post-training quant model](https://paddle-inference-dist.cdn.bcebos.com/quantizaiton/quant_post_models/mobilenetv2_quant_post.tgz)以及其对应的原始的FP32模型[mobilenetv2 fp32](https://paddle-inference-dist.cdn.bcebos.com/quantizaiton/fp32_models/mobilenetv2.tgz)。如果用户要转化部署自己的模型，请根据下面2.1, 2.2的步骤产出量化模型。

#### 2.1 量化训练

量化训练流程可以参考 [分类模型的量化训练流程](https://paddlepaddle.github.io/PaddleSlim/tutorials/quant_aware_demo/)

**量化训练过程中config参数：**
- **quantize_op_types:** 目前CPU上量化支持的算子为 `depthwise_conv2d`, `conv2d`, `mul`, `matmul`, `transpose2`, `reshape2`, `pool2d`, `scale`, `concat`。但是在量化训练阶段插入fake_quantize/fake_dequantize算子时，只需在前四种op前后插入fake_quantize/fake_dequantize 算子，因为后面四种算子 `transpose2`, `reshape2`, `pool2d`, `scale`, `concat`的scales将从其他op的`out_threshold`属性获取。所以，在使用PaddleSlim量化训练时，只可以对 `depthwise_conv2d`, `conv2d`, `mul`, `matmul`进行量化，不支持其他op。
- **其他参数:** 请参考 [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)

#### 2.2 离线量化

离线量化模型产出可以参考[分类模型的静态离线量化流程](https://paddlepaddle.github.io/PaddleSlim/tutorials/quant_post_demo/#_1)

在使用PaddleSlim离线量化时，只可以对 `depthwise_conv2d`, `conv2d`, `mul`, `matmul`进行量化，不支持其他op。

## 3. 转化产出的量化模型为DNNL优化后的INT8模型

为了部署在CPU上，我们将保存的quant模型，通过一个转化脚本，移除fake_quantize/fake_dequantize op，进行算子融合和优化并且转化为INT8模型。

脚本在官网的位置为[save_quant_model.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py)。

复制脚本到本样例所在目录(`/PATH_TO_PaddleSlim/demo/mkldnn_quant/`)，并执行如下命令：
```
python save_quant_model.py --quant_model_path=/PATH/TO/SAVE/FLOAT32/QUANT/MODEL --int8_model_save_path=/PATH/TO/SAVE/INT8/MODEL
```
**参数说明：**
- **quant_model_path:** 为输入参数，必填。为量化训练产出的quant模型。
- **int8_model_save_path:** 将quant模型经过DNNL优化量化后保存的最终INT8模型输出路径。注意：quant_model_path必须传入PaddleSlim量化产出的含有fake_quant/fake_dequant ops的quant模型。
- **ops_to_quantize:** 以逗号隔开的指定的需要量化的op类型列表。可选，默认为空，空表示量化所有可量化的op。目前，对于Benchmark中列出的图像分类和自然语言处理模型中，量化所有可量化的op可以获得最好的精度和性能，因此建议用户不设置这个参数。
- **--op_ids_to_skip:** 以逗号隔开的op id号列表，可选，默认为空。这个列表中的op号将不量化，采用FP32类型。要获取特定op的ID，请先使用`--debug`选项运行脚本，并打开生成的文件`int8_<number>_cpu_quantize_placement_pass.dot`，找出不需量化的op, ID号在Op名称后面的括号中。
- **--debug:** 添加此选项可在每个转换步骤之后生成一系列包含模型图的* .dot文件。 有关DOT格式的说明，请参见[DOT](https://graphviz.gitlab.io/_pages/doc/info/lang.html)。要打开`* .dot`文件，请使用系统上可用的任何Graphviz工具（例如Linux上的`xdot`工具或Windows上的`dot`工具有关文档，请参见[Graphviz](http://www.graphviz.org/documentation/)。
- **注意：**
  - 目前支持DNNL量化的op列表是`conv2d`, `depthwise_conv2d`, `fc`, `matmul`, `pool2d`, `reshape2`, `transpose2`,`scale`, `concat`。
  - 如果设置 `--op_ids_to_skip`,只需要传入所有量化op中想要保持FP32类型的op ID号即可。
  - 有时量化全部op不一定得到最优性能。例如：如果一个op是单个的INT8 op, 之前和之后的op都为float32 op,那么为了量化这个op，需要先做quantize，然后运行INT8 op, 再dequantize, 这样可能导致最终性能不如保持该op为fp32 op。如果用户使用默认设置性能较差，可以观察这个模型是否有单独的INT8 op，选出不同的`ops_to_quantize`组合，也可以通过`--op_ids_to_skip`排除部分可量化op ID，多运行几次获得最佳设置。

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
- 为什么将数据集转化为二进制文件？因为paddle中的数据预处理（resize, crop等）都使用pythong.Image模块进行，训练出的模型也是基于Python预处理的图片，但是我们发现Python测试性能开销很大，导致预测性能下降。为了获得良好性能，在量化模型预测阶段，我们决定使用C++测试，而C++只支持Open-CV等库，Paddle不建议使用外部库，因此我们使用Python将图片预处理然后放入二进制文件，再在C++测试中读出。用户根据自己的需要，可以更改C++测试以直接读数据并预处理，精度不会有太大下降。我们还提供了python测试`sample_tester.py`作为参考，与C++测试`sample_tester.cc`相比，用户可以看到Python测试更大的性能开销。

### 4.2 部署预测

#### 部署前提

- 用户可以通过在命令行红输入`lscpu`查看本机支持指令。
- 在支持`avx512_vnni`的CPU服务器上，INT8精度和性能最高，如：Casecade Lake, Model name: Intel(R) Xeon(R) Gold X2XX，INT8性能提升为FP32模型的3~3.7倍
- 在支持`avx512`但是不支持`avx512_vnni`的CPU服务器上，如：SkyLake, Model name：Intel(R) Xeon(R) Gold X1XX，INT8性能为FP32性能的1.5倍左右。

#### 准备预测推理库

用户可以从源码编译Paddle推理库，也可以直接下载推理库。
- 用户可以从Paddle源码编译Paddle推理库，参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12)，使用release/2.0以上版本。

- 用户也可以从Paddle官网下载发布的[预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。请选择`ubuntu14.04_cpu_avx_mkl` 最新发布版或者develop版。

你可以将准备好的预测库解压并重命名为fluid_inference，放在当前目录下(`/PATH_TO_PaddleSlim/demo/mkldnn_quant/`)。或者在cmake时通过设置PADDLE_ROOT来指定Paddle预测库的位置。

#### 编译应用

样例所在目录为PaddleSlim下`demo/mkldnn_quant/`,样例`sample_tester.cc`和编译所需`cmake`文件夹都在这个目录下。
```
cd /PATH/TO/PaddleSlim
cd demo/mkldnn_quant/
mkdir build
cd build
cmake -DPADDLE_ROOT=$PADDLE_ROOT ..
make -j
```
如果你从官网下载解压了[预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)到当前目录下，这里`-DPADDLE_ROOT`可以不设置，因为`-DPADDLE_ROOT`默认位置`demo/mkldnn_quant/fluid_inference`

#### 运行测试
```
# Bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
# Turbo Boost could be set to OFF using the command
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# In the file run.sh, set `MODEL_DIR` to `/PATH/TO/FLOAT32/MODEL`或者`/PATH/TO/SAVE/INT8/MODEL`
# In the file run.sh, set `DATA_FILE` to `/PATH/TO/SAVE/BINARY/FILE`
# For 1 thread performance:
./run.sh
# For 20 thread performance:
./run.sh -1 20
```

运行时需要配置以下参数：
- **infer_model:** 模型所在目录，注意模型参数当前必须是分开保存成多个文件的。可以设置为`PATH/TO/SAVE/INT8/MODEL`, `PATH/TO/SAVE/FLOAT32/MODEL`。无默认值。
- **infer_data:** 测试数据文件所在路径。注意需要是经`full_ILSVRC2012_val_preprocess`转化后的binary文件。
- **batch_size:** 预测batch size大小。默认值为50。
- **iterations:** batches迭代数。默认为0，0表示预测infer_data中所有batches (image numbers/batch_size)
- **num_threads:** 预测使用CPU 线程数，默认为单核一个线程。
- **with_accuracy_layer:** 模型为包含精度计算层的测试模型还是不包含精度计算层的预测模型，默认为true。
- **use_analysis** 是否使用`paddle::AnalysisConfig`对模型优化、融合(fuse)，加速。默认为false

你可以直接修改`/PATH_TO_PaddleSlim/demo/mkldnn_quant/`目录下的`run.sh`中的MODEL_DIR和DATA_DIR，即可执行`./run.sh`进行CPU预测。

### 4.3 用户编写自己的测试：

如果用户编写自己的测试：
1. 测试INT8模型
    如果用户测试转化好的INT8模型，使用 `paddle::NativeConfig` 即可测试。在demo中，设置`use_analysis`为`false`。
2. 测试FP32模型
   如果用户要测试PF32模型，使用`paddle::AnalysisConfig`对原始FP32模型先优化（fuses等）再测试。在样例中，直接设置`use_analysis`为`true`。AnalysisConfig设置如下：
```
static void SetConfig(paddle::AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);  // 必须。表示需要测试的模型
  cfg->DisableGpu();      // 必须。部署在CPU上预测，必须Disablegpu
  cfg->EnableMKLDNN();  //必须。表示使用MKLDNN算子，将比 native 快
  cfg->SwitchIrOptim();   // 如果传入FP32原始，这个配置设置为true将优化加速模型
  cfg->SetCpuMathLibraryNumThreads(FLAGS_num_threads);  //非必须。默认设置为1。表示多线程运行
}
```
- 在我们提供的样例中，只要设置`use_analysis`为true并且`infer_model`传入原始FP32模型，AnalysisConfig的上述设置将被执行，传入的FP32模型将被DNNL优化加速（包括fuses等）。
- 如果infer_model传入INT8模型，则 `use_analysis`将不起任何作用，因为INT8模型已经被优化量化。
- 如果infer_model传入PaddleSlim产出的quant模型，`use_analysis`即使设置为true不起作用，因为quant模型包含fake_quantize/fake_dequantize ops,无法fuse,无法优化。

## 5. 精度和性能数据
INT8模型精度和性能结果参考[CPU部署预测INT8模型的精度和性能](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/docs/zh_cn/tutorials/image_classification_mkldnn_quant_tutorial.md)

## FAQ

- 自然语言处理模型在CPU上的部署和预测参考样例[ERNIE 模型 QUANT INT8 精度与性能复现](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c++/ernie/mkldnn)
- 具体DNNL量化原理可以查看[SLIM Quant for INT8 DNNL](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/README.md)。

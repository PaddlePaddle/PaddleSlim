# PaddleSlim量化模型的TensorRT预测

本教程将介绍使用TensortRT部署PaddleSlim量化得到的模型的详细步骤。


## 1. 准备环境

### 1.1 下载或者编译支持TensorRT的Paddle预测库

* 有2种方式获取Paddle预测库，下面进行详细介绍。

#### 1.1.1 预测库源码编译
* 如果希望获取最新预测库特性，可以从Paddle github上克隆最新代码，源码编译预测库。
* 可以参考[Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)的说明，从github上获取Paddle代码，然后进行编译，生成最新的预测库。使用git获取代码方法如下。

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```
* 在[Nvidia官网](https://developer.nvidia.com/TensorRT)下载TensorRT并解压, 本示例以TensorRT 6.0为例。

* 进入Paddle目录后，编译方法如下。

```shell
rm -rf build
mkdir build
cd build

cmake  .. \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DTENSORRT_ROOT=TensorRT-6.0.1.5 \
    -DFLUID_INFERENCE_INSTALL_DIR=LIB_ROOT \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
make -j
make inference_lib_dist
```

其中`DFLUID_INFERENCE_INSTALL_DIR`代表编译完成后预测库生成的地址，`DTENSORRT_ROOT`代表下载解压后的TensorRT路径。

更多编译参数选项可以参考Paddle C++预测库官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


* 编译完成之后，可以在`LIB_ROOT`路径下看到生成了以下文件及文件夹。

```
LIB_ROOT/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

其中`paddle`就是之后进行TensorRT预测时所需的Paddle库，`version.txt`中包含当前预测库的版本信息。

#### 1.2.2 直接下载安装

* [Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)上提供了不同cuda版本的Linux预测库，可以在官网查看并选择带有TensorRT的预测库版本。

* 下载之后使用下面的方法解压。

```
tar -xf fluid_inference.tgz
```

最终会在当前的文件夹中生成`fluid_inference/`的子文件夹。


## 2 开始运行

### 2.1 将模型导出为inference model

* 可以参考[量化训练教程](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/quant_aware_tutorial.html#id9)，在训练完成后导出inference model。

```
inference/
|-- model
|-- params
```


### 2.2 编译TensorRT预测demo

* 编译命令如下，其中Paddle, TensorRT地址需要换成自己机器上的实际地址。


```shell
sh tools/build.sh
```

具体地，`tools/build.sh`中内容如下。

```shell
PADDLE_LIB_PATH=trt_inference # change to your path
USE_GPU=ON
USE_MKL=ON
USE_TRT=ON
TENSORRT_INCLUDE_DIR=TensorRT-6.0.1.5/include # change to your path
TENSORRT_LIB_DIR=TensorRT-6.0.1.5/lib # change to your path

if [ $USE_GPU -eq ON ]; then
  export CUDA_LIB=`find /usr/local -name libcudart.so`
fi
BUILD=build
mkdir -p $BUILD
cd $BUILD
cmake .. \
      -DPADDLE_LIB=${PADDLE_LIB_PATH} \
      -DWITH_GPU=${USE_GPU} \
      -DWITH_MKL=${USE_MKL} \
      -DCUDA_LIB=${CUDA_LIB} \
      -DUSE_TENSORRT=${USE_TRT} \
      -DTENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR} \
      -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR}
make -j4
```

`PADDLE_LIB_PATH`为下载(`fluid_inference`文件夹)或者编译生成的Paddle预测库地址(`build/fluid_inference_install_dir`文件夹)；`TENSORRT_INCLUDE_DIR`和`TENSORRT_LIB_DIR`分别代表TensorRT的include和lib目录路径。


* 编译完成之后，会在`build`文件夹下生成可执行文件。


### 2.3 数据预处理转化

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
- 为什么将数据集转化为二进制文件？因为paddle中的数据预处理（resize, crop等）都使用pythong.Image模块进行，训练出的模型也是基于Python预处理的图片，但是我们发现Python测试性能开销很大，导致预测性能下降。为了获得良好性能，在量化模型预测阶段，我们需要使用C++测试，而C++只支持Open-CV等库，Paddle不建议使用外部库，因此我们使用Python将图片预处理然后放入二进制文件，再在C++测试中读出。用户根据自己的需要，可以更改C++测试以直接读数据并预处理，精度不会有太大下降。

### 2.4 部署预测


### 运行demo
* 执行以下命令，完成对一个模型的TensorRT速度评测。

```shell
sh tools/run_benchmark.sh
```

其中`MODEL_DIR`和`DATA_FILE`分别代表模型文件和数据文件, 需要在预测时替换为自己实际要用的地址。

* 执行以下命令，完成对一个模型的TensorRT精度评测。

```shell
sh tools/test_acc.sh
```

同上，在预测时需要将其中路径替换为自己实际要用的地址。

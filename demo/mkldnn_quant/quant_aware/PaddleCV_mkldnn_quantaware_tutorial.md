# Image classification models fixed-point quantization tutorial

## Overview

Quantization is an important method for model compression and inference performance improvement. PaddleSlim supports two quantization strategies: `post` and `aware`. This tutorial present the way to deploy and inference both post-training models and quantization-aware traing models on CPUS. On Intel (R) Cascade Lake class CPU machines, 8-bits quantization, graph optimizations and DNNL acceleration yields performance of a quantized model up to 4 times better than of an original FP32 model. Currently, we support quantizable operators `conv2d`, `depthwise_conv2d`, `mul`, `matmul`, `transpose2`, `reshape2`, `pool2d`. DNNL optimizations consist mainly of operator fusing passes which simplify the model graph greatly, further improving the performance, including `batch_norm`, `relu`, `brelu`, `elementwise_add`, etc. After quantization and fuses, INT8 models performance will be greatly improved. For details about DNNL optimization users can refer to [SLIM QAT for DNNL INT8](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/QAT_mkldnn_int8_readme.md)

**Note**:

- PaddlePaddle in version 1.8 or higher is required.
- DNNL and MKL are required.
- INT8 accuracy and performance are best on CPU servers supporting AVX512 VNNI extension (e.g. CLX class Intel processors). A linux server supports AVX512 VNNI instructions if the output of the command `lscpu` contains the `avx512_vnni` entry in the Flags section. AVX512 VNNI support on Windows can be checked using the coreinfo tool.

## 1. Deployment

#### 1.1 Prepare Inference library

Users can build Paddle Inference library from source or directly download from the official website.

- Build Paddle Inference from source, please refer to [Build from source](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12)  options as follows:
>**CMake options enabling for quantization demo**

| Option        | Value           | Description  |
| ------------- |:-------------:|:-----:|
| ON_INFER      | ON | Turn on for inference|
| WITH_MKL      | ON | Turn on MKL because we use MKL in following test|
| WITH_MKLDNN   | ON | Turn on MKLDNN because we use MKLDNN in following test|

- Download from [fluid_inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html). Please download`ubuntu14.04_cpu_avx_mkl` newest release or develop version

You can rename your prepared library as `fluid_inference`, put it under the same folder of the test. During compling, the test will locate inference library at `./fluid_inference`. You can also set PADDLE_ROOT to your saved inference library when you compile the test.


### 1.2 Prepare PaddleSlim
```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```
### 1.3 Use paddle and slim in sample code
You can use paddle and paddleslim as follows:
```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. Training and converting fp32 model to fp32 qat model
Users can download pretrained fp32 models at [Download pre-trained models](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)

We provide a script to insert `fake_quantize`/`fake_dequantize` ops, train a few iterations and then save the float32 QAT model. Run as follows
```
cd PaddleSlim/demo/mkldnn_quant/quant_aware/
python train_image_classification.py --model=ResNet50 --pretrained_model=path/to/ResNet50/pretrained --data=imagenet --data_dir=/PATH/TO/ILSVRC2012/  --batch_size=32 --num_epochs=1 --save_float32_qat_dir=/PATH/TO/FLOAT32/QAT/MODEL
```
Available options in the above command and their descriptions are as follows:
- **model:** Model name. The model name should be one of these: `['MobileNet', 'MobileNetV2', 'PVANet', 'ResNet34', 'ResNet50', 'ResNet50_vd']`. Default value: "ResNet50"
- **pretrained_model:** Path to pre-trained model. Default value: None
- **data:** Dataset. Default value: imagenet
- **data_dir:** Path to the dataset. Default value: None
- **batch_size:** Number of training batch size. Default value: 128
- **num_epochs:** Number of training epoches. Default value: 1
- **save_float32_qat_dir:** Path to saved qat_float32 model. Default value: `./quantization_models/`
- **config_file:** Path to training config file. Default value is `./config.yaml`

If the user needs to change the quantization strategy, modify `config.yaml`. We sugggest the following configuration to obtain the best accuracy.
```
config = {
         'weight_quantize_type': 'channel_wise_abs_max',
         'activation_quantize_type': 'moving_average_abs_max',
         'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d', 'matmul']
     }
```
**`config.yaml` parameters description：**
- **quantize_op_types:** Now we support quantizable ops including `depthwise_conv2d`, `mul`, `conv2d`, `matmul`, `transpose2`, `reshape2`, `pool2d`. During the period of training with fake quantize/dequantize op, we only need to insert fake quantize/dequantize ops around 4 types of ops:`depthwise_conv2d`, `mul`, `conv2d`, `matmul`. That is because other quantizable ops `transpose2`, `reshape2`, `pool2d`, input scale and output scale are the same, and the scales could be achieved through `out_threshold` attribute of the op, there is no need to insert fake quantize/dequantize ops for these ops. Hereby, we set `'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d', 'matmul']`
- **Other parameters:** Please refer [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)

**Note:**
- To modify the program for quantization, some training options have to be disabled. Add following lines in the training code to disable these options::
```
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
```
- In `train_image_classification.py`, `paddleslim.quant.convert` is used to change the order of `fake_quantize`/`fake_dequantize` ops in Program. In addition, `paddleslim.quant.convert ` will change the operator parameters values to the values within the range of quantized `int8_t`, but the data type is still `float32`. This is the fp32 qat model we need, the default saving location is `./quantization_models/`.

## 3. Convert fp32 qat model to INT8 model
The model saved after training in the previous step is the float32 qat model. We have to remove the `fake_quantize`/`fake_dequantize` ops, and fully convert it into INT8 model. Go to the Paddle directory and run

```
python Paddle/python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=/PATH/TO/FLOAT32/QAT/MODEL --int8_model_save_path=/PATH/TO/SAVE/INT8/MODEL --fp32_model_save_path=/PATH/TO/SAVE/FLOAT32/MODEL --ops_to_quantize="conv2d,pool2d"
```
**Available options in the above command：**
- **qat_model_path:** The path where we save qat float32 model after training.
- **fp32_model_save_path:** Optional. If set, meaning the path to the saved optimized fp32 model
- **int8_model_save_path:** Optional. If set, meaning the path to the saved optimized and quantized INT8 model
- **ops_to_quantize:**  A comma-separated list of operator types to quantize. If the option is not used, an attempt to quantize all quantizable operators will be made, and in that case only quantizable operators which have quantization scales provided in the QAT model will be quantized. When deciding which operators to put on the list, the following have to be considered:
  - Only operators which support quantization will be taken into account.
  - All the quantizable operators from the list, which are present in the model, must have quantization scales provided in the model. Otherwise, quantization of the operator will be skipped with a message saying which variable is missing a quantization scale.
  - Sometimes it may be suboptimal to quantize all quantizable operators in the model (cf. Notes in the Gathering scales section above). To find the optimal configuration for this option, user can run benchmark a few times with different lists of quantized operators present in the model and compare the results. For Image Classification models mentioned above the list usually comprises of conv2d and pool2d operators.

## 4. Inference test

### 4.1 Data preprocessing
To run the inference test, the data needs to be converted to binary first. Run the following script without any pramaters allows you to transform the complete ILSVRC2012_val_data set to bianary file. Use `local` parameter to transform your own data. Go to Paddle directory and run:
```
python Paddle/paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py --local --data_dir=/PATH/TO/USER/DATASET/  --output_file=/PATH/TO/SAVE/BINARY/FILE
```

Available options in the above command and their descriptions are as follows:
- No parameters set. The script will download the ILSVRC2012_img_val data from server and convert it into a binary file.
- **local:** once set, the script will process user data.
- **data_dir:** Path to the user data directory.
- **label_list:** set the image path-image category list, similar to `val_list.txt`.
- **output_file:** Path of the generated bin file.
- **data_dim:** The length and width of the preprocessed image. Default is 224.

The user's own data set directory structure should be as follows:
```
imagenet_user
├── val
│ ├── ILSVRC2012_val_00000001.jpg
│ ├── ILSVRC2012_val_00000002.jpg
| | ── ...
└── val_list.txt
```
Among them, the content of val_list.txt should be as follows:
```
val/ILSVRC2012_val_00000001.jpg 0
val/ILSVRC2012_val_00000002.jpg 0
```

note:
- The reason for converting the dataset into a binary file is performance and independence of external C++ libraries. Image data requires preprocessing like resizing, cropping, etc. and it can be easily achieved using python Image module, both for training and inference. However, the performance of python tests is lower than of C++ tests, hence the decision to latter for the quantitative model inference tests are made. While effective image processing in C++ requires linking to external libraries like Open-CV, to avoid adding new dependencies to Paddle we use Python to preprocess the image data and put the result in a binary file. Then, the binary dataset is read into the C++ test. A python test sample_tester.py is there for reference, so users can observe its performance overhead compared to the C++ test sample_tester.cc.

### 4.2 Compile and run inference
####  Build the application
Run following commnd under the test directory:
```
cd PaddleSlim/demo/mkldnn_quant/quant_aware/
mkdir build
cd build
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=$PADDLE_ROOT -DUSE_PROFILER=OFF ..
make -j
```
#### Run the test
```
# Bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
# Turbo Boost was set to OFF using the command
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# In the file run.sh, set `MODEL_DIR` to `/PATH/TO/SAVE/INT8/MODEL`, it can be `/PATH/TO/SAVE/FLOAT32/MODEL` or `/PATH/TO/FLOAT32/QAT/MODEL`
# In the file run.sh, set `DATA_FILE` to `/PATH/TO/SAVE/BINARY/FILE`
# For 1 thread performance:
./run.sh
# For 20 thread performance:
./run.sh -1 20
```

Available options in script `run.sh` and their descriptions are as follows:
- **infer_model：** The path of the model. Note that the model parameters must be saved separately into multiple files. Default value None.
- **infer_data：** The path of test data. Note that it must be a binary file converted by `full_ILSVRC2012_val_preprocess`.
- **batch_size：** Inference batch size. Default value 50.
- **iterations：** Inference iteration number. Default value is 0, which means predicting all batches (image numbers / batch size) in infer_data
- **num_threads：** The number of CPU threads to be used. Default value 1.
- **with_accuracy_layer：** The model is with accuracy layer or not. Default value is false.
- **use_profile：** Do profiling or not. Default value is false. If you want to set `use_profile` to `true`, users need to build Paddle with `-DWITH_PROFILER=ON` and build test application with `-DUSE_PROFILER=ON` in advance.

## 5. Accuracy and Performance benchmark

This section contain QAT2 DNNL accuracy and performance benchmark results measured on two server
* Intel(R) Xeon(R) Gold 6271 (with AVX512 VNNI support),
* Intel(R) Xeon(R) Gold 6148.

#### 5.1 Accuracy

>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

>**Intel(R) Xeon(R) Gold 6148**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.85%         |   0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.08%         |   0.18%   |       90.56%       |         90.66%         |  +0.10%   |
|  ResNet101   |       77.50%       |         77.51%         |   0.01%   |       93.58%       |         93.50%         |  -0.08%   |
|   ResNet50   |       76.63%       |         76.55%         |  -0.08%   |       93.10%       |         92.96%         |  -0.14%   |
|    VGG16     |       72.08%       |         71.72%         |  -0.36%   |       90.63%       |         89.75%         |  -0.88%   |
|    VGG19     |       72.57%       |         72.08%         |  -0.49%   |       90.84%       |         90.11%         |  -0.73%   |

#### 5.2 Performance

Image classification models performance was measured using a single thread. The setting is included in the benchmark reproduction commands below.

>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      74.36      |       210.68        |       2.83        |
| MobileNet-V2 |      89.59      |       186.55        |       2.08        |
|  ResNet101   |      7.21       |        26.41        |       3.67        |
|   ResNet50   |      13.23      |        48.89        |       3.70        |
|    VGG16     |      3.49       |        10.11        |       2.90        |
|    VGG19     |      2.84       |        8.69         |       3.06        |

>**Intel(R) Xeon(R) Gold 6148**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      75.23      |       111.15        |       1.48        |
| MobileNet-V2 |      86.65      |       127.21        |       1.47        |
|  ResNet101   |      6.61       |        10.60        |       1.60        |
|   ResNet50   |      12.42      |        19.74        |       1.59        |
|    VGG16     |      3.31       |        4.74         |       1.43        |
|    VGG19     |      2.68       |        3.91         |       1.46        |

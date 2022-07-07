# 自动化压缩工具ACT（Auto Compression Toolkit）

## 简介
PaddleSlim推出全新自动化压缩工具（ACT），旨在通过Source-Free的方式，自动对预测模型进行压缩，压缩后模型可直接部署应用。ACT自动化压缩工具主要特性如下：
- **『更便捷』**：开发者无需了解或修改模型源码，直接使用导出的预测模型进行压缩；
- **『更智能』**：开发者简单配置即可启动压缩，ACT工具会自动优化得到最好预测模型；
- **『更丰富』**：ACT中提供了量化训练、蒸馏、结构化剪枝、非结构化剪枝、多种离线量化方法及超参搜索等等，可任意搭配使用。


## 环境准备

- 安装PaddlePaddle >= 2.3 （从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- 安装PaddleSlim >=2.3

（1）安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

（2）安装paddleslim：
```shell
pip install paddleslim
```

## 快速上手

- 1.准备模型及数据集

```shell
# 下载MobileNet预测模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar -xf MobileNetV1_infer.tar
# 下载ImageNet小型数据集
wget https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz
tar -xf ILSVRC2012_data_demo.tar.gz
```

- 2.运行

```python
# 导入依赖包
import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddleslim.auto_compression import AutoCompression
paddle.enable_static()
# 定义DataSet
class ImageNetDataset(DatasetFolder):
    def __init__(self, path, image_size=224):
        super(ImageNetDataset, self).__init__(path)
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size), transforms.Transpose(),
            normalize
        ])

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.samples)

# 定义DataLoader
train_dataset = ImageNetDataset("./ILSVRC2012_data_demo/ILSVRC2012/train/")
image = paddle.static.data(
    name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
train_loader = paddle.io.DataLoader(train_dataset, feed_list=[image], batch_size=32, return_list=False)
# 开始自动压缩
ac = AutoCompression(
    model_dir="./MobileNetV1_infer",
    model_filename="inference.pdmodel",
    params_filename="inference.pdiparams",
    save_dir="MobileNetV1_quant",
    config={'Quantization': {}, "HyperParameterOptimization": {'ptq_algo': ['avg'], 'max_quant_count': 3}},
    train_dataloader=train_loader,
    eval_dataloader=train_loader)
ac.compress()
```

- 3.测试精度

测试压缩前模型的精度:
```shell
CUDA_VISIBLE_DEVICES=0 python ./image_classification/eval.py
### Eval Top1: 0.7171724759615384
```

测试量化模型的精度:
```shell
CUDA_VISIBLE_DEVICES=0 python ./image_classification/eval.py --model_dir='MobileNetV1_quant'
### Eval Top1: 0.7166466346153846
```

量化后模型的精度相比量化前的模型几乎精度无损，由于是使用的超参搜索的方法来选择的量化参数，所以每次运行得到的量化模型精度会有些许波动。

- 4.推理速度测试
量化模型速度的测试依赖推理库的支持，所以确保安装的是带有TensorRT的PaddlePaddle。以下示例和展示的测试结果是基于Tesla V100、CUDA 10.2、python3.7得到的。

使用以下指令查看本地cuda版本，并且在[下载链接](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)中下载对应cuda版本和对应python版本的paddlepaddle安装包。
```shell
cat /usr/local/cuda/version.txt ### CUDA Version 10.2.89
### 10.2.89 为cuda版本号，可以根据这个版本号选择需要安装的带有TensorRT的PaddlePaddle安装包。
```

安装下载的whl包：
```
### 这里通过wget下载到的是python3.7、cuda10.2的PaddlePaddle安装包，若您的环境和示例环境不同，请依赖您自己机器的环境下载对应的安装包，否则运行示例代码会报错。
wget https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_x86_64.whl
pip install paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_x86_64.whl --force-reinstall
```

测试FP32模型的速度
```
python ./image_classification/infer.py
### using tensorrt FP32	batch size: 1 time(ms): 0.6140608787536621
```

测试FP16模型的速度
```
python ./image_classification/infer.py --use_fp16=True
### using tensorrt FP16	batch size: 1 time(ms): 0.5795984268188477
```

测试INT8模型的速度
```
python ./image_classification/infer.py --model_dir=./MobileNetV1_quant/ --use_int8=True
### using tensorrt INT8 batch size: 1 time(ms): 0.5213963985443115
```

**提示：**
- DataLoader传入的数据集是待压缩模型所用的数据集，DataLoader继承自`paddle.io.DataLoader`。可以直接使用模型套件中的DataLoader，或者根据[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)自定义所需要的DataLoader。
- 自动化压缩Config中定义量化、蒸馏、剪枝等压缩算法会合并执行，压缩策略有：量化+蒸馏，剪枝+蒸馏等等。示例中选择的配置为离线量化超参搜索。
- 如果要压缩的模型参数是存储在各自分离的文件中，需要先通过[convert.py](./convert.py) 脚本将其保存成一个单独的二进制文件。

## 应用示例

#### [图像分类](./image_classification)

#### [目标检测](./detection)

#### [语义分割](./semantic_segmentation)

#### [NLP](./nlp)

#### X2Paddle

- [PyTorch YOLOv5](./pytorch_yolov5)
- [HuggingFace](./pytorch_huggingface)
- [TensorFlow MobileNet](./tensorflow_mobilenet)

#### 即将发布
- [ ] 更多自动化压缩应用示例

## 其他

- ACT可以自动处理常见的预测模型，如果有更特殊的改造需求，可以参考[ACT超参配置教程](./hyperparameter_tutorial.md)来进行单独配置压缩策略。

- 如果你发现任何关于ACT自动化压缩工具的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSlim/issues)给我们提issues。同时欢迎贡献更多优秀模型，共建开源生态。

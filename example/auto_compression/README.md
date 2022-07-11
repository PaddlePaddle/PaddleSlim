# 模型自动化压缩工具ACT（Auto Compression Toolkit）

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/Paddle?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleSlim?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleSlim?color=3af"></a>
    <a href="https://pypi.org/project/PaddleSlim/"><img src="https://img.shields.io/pypi/dm/PaddleSlim?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleSlim?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSlim/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleSlim?color=ccf"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#模型压缩效果Benchmark> Benchmark </a> |
  <a href=#环境准备> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#进阶使用> 进阶使用 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

## **简介**

PaddleSlim推出全新自动化压缩工具（Auto Compression Toolkit, ACT），旨在通过Source-Free的方式，自动对预测模型进行压缩，压缩后模型可直接部署应用。

## **News** 📢

* 🎉 2022.7.6 [**PaddleSlim v2.3.0**](https://github.com/PaddlePaddle/PaddleSlim/releases/tag/v2.3.0)全新发布！目前已经在图像分类、目标检测、图像分割、NLP等20多个模型验证正向效果。
* 🔥 2022.7.14 晚 20:30，PaddleSlim自动压缩天使用户沟通会。与开发者共同探讨模型压缩痛点问题，欢迎大家扫码报名入群获取会议链接。

  <div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/178181077-57a3a631-f495-4821-878d-ef5e74981718.jpg"  width = "150" height = "150" />
  </div>

## **特性**

- <a href=#解耦训练代码>  **🚀『解耦训练代码』** </a>：开发者无需了解或修改模型源码，直接使用导出的预测模型进行压缩；
- <a href=#全流程自动优化> **🎛️『全流程自动优化』** </a>：开发者简单配置即可启动压缩，ACT工具会自动优化得到最好预测模型；
- <a href=#支持丰富压缩算法> **📦『支持丰富压缩算法』** </a>：ACT中提供了量化训练、蒸馏、结构化剪枝、非结构化剪枝、多种离线量化方法及超参搜索等等，可任意搭配使用

### **ACT核心思想**

相比于传统手工压缩，自动化压缩的“自动”主要体现在4个方面：解耦训练代码、离线量化超参搜索、算法

<p align="center">
  <img src="https://user-images.githubusercontent.com/23690325/178102488-9f09e991-bfd6-4827-8641-849d9c3fa83c.png" align="middle"  width="800" />
</p>

### **模型压缩效果示例**

ACT相比传统的模型压缩方法，

- 代码量减少 50% 以上
- 压缩精度与手工压缩基本持平。在 PP-YOLOE 模型上，效果还优于手动压缩，
- 自动化压缩后的推理性能收益与手工压缩持平，相比压缩前，推理速度可以提升1.4~7.1倍。

<p align="center">
  <img src="https://user-images.githubusercontent.com/23690325/178102623-6de25af1-eec8-4825-bb15-4dad5bee7c9c.png" align="middle"  width="800" />
</p>

### **模型压缩效果Benchmark**

<font size=5>  </font>

<font size=0.5>

| 模型类型                            | model name                   | 压缩前<br/>精度(Top1 Acc %) | 压缩后<br/>精度(Top1 Acc %) | 压缩前<br/>推理时延（ms） | 压缩后<br/>推理时延（ms） | 推理<br/>加速比 | 芯片                |
| ------------------------------- | ---------------------------- | ---------------------- | ---------------------- | ---------------- | ---------------- | ---------- | ----------------- |
| [图像分类](./image_classification)  | MobileNetV1                  | 70.90                  | 70.57                  | 33.15            | 13.64            | **2.43**   | SDM865（晓龙865）     |
| [图像分类](./image_classification)  | ShuffleNetV2_x1_0            | 68.65                  | 68.32                  | 10.43            | 5.51             | **1.89**   | SDM865（晓龙865）     |
| [图像分类](./image_classification)  | SqueezeNet1_0_infer          | 59.60                  | 59.45                  | 35.98            | 16.96            | **2.12**   | SDM865（晓龙865）     |
| [图像分类](./image_classification)  | PPLCNetV2_base               | 76.86                  | 76.43                  | 36.50            | 15.79            | **2.31**   | SDM865（晓龙865）     |
| [图像分类](./image_classification)  | ResNet50_vd                  | 79.12                  | 78.74                  | 3.19             | 0.92             | **3.47**   | NVIDIA Tesla T4   |
| [语义分割](./semantic_segmentation) | PPHGNet_tiny                 | 79.59                  | 79.20                  | 2.82             | 0.98             | **2.88**   | NVIDIA Tesla T4   |
| [语义分割](./semantic_segmentation) | PP-HumanSeg-Lite             | 92.87                  | 92.35                  | 56.36            | 37.71            | **1.49**   | SDM710            |
| [语义分割](./semantic_segmentation) | PP-LiteSeg                   | 77.04                  | 76.93                  | 1.43             | 1.16             | **1.23**   | NVIDIA Tesla T4   |
| [语义分割](./semantic_segmentation) | HRNet                        | 78.97                  | 78.90                  | 8.19             | 5.81             | **1.41**   | NVIDIA Tesla T4   |
| [语义分割](./semantic_segmentation) | UNet                         | 65.00                  | 64.93                  | 15.29            | 10.23            | **1.49**   | NVIDIA Tesla T4   |
| NLP                             | PP-MiniLM                    | 72.81                 | 72.44                 | 128.01           | 17.97            | **7.12**   | NVIDIA Tesla T4   |
| NLP                             | ERNIE 3.0-Medium             | 73.09                 | 72.40                 | 29.25(fp16)      | 19.61            | **1.49**   | NVIDIA Tesla T4   |
| [目标检测](./detection)             | YOLOv5s<br/>(PyTorch)        | 37.40                  | 36.9                   | 5.95             | 1.87             | **3.18**   | NVIDIA Tesla T4   |
| [目标检测](./detection)             | PP-YOLOE-l                   | 50.9                   | 50.6                   | 11.2             | 6.7              | **1.67**   | NVIDIA Tesla V100 |
| [图像分类](./image_classification)  | MobileNetV1<br/>(TensorFlow) | 71.0                   | 70.22                  | 30.45            | 15.86            |  **1.92**  | SDMM865（晓龙865）     |  

- 备注：目标检测精度指标为mAP（0.5:0.95）精度测量结果。图像分割精度指标为IoU精度测量结果。
- 更多飞桨模型应用示例及Benchmark可以参考：[图像分类](./image_classification)，[目标检测](./detection)，[语义分割](./semantic_segmentation)，[自然语言处理](./nlp)
- 更多其它框架应用示例及Benchmark可以参考：[YOLOv5(PyTorch)](./pytorch_yolov5)，[HuggingFace(PyTorch)](./pytorch_huggingface)，[MobileNet(TensorFlow)](./tensorflow_mobilenet)

## **环境准备**

- 安装PaddlePaddle >= 2.3.1：（可以参考[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）

  ```shell
  # CPU
  pip install paddlepaddle --upgrade
  # GPU
  pip install paddlepaddle-gpu --upgrade
  ```

- 安装PaddleSlim >=2.3.0：

  ```shell
  pip install paddleslim
  ```

## **快速开始**

- **1. 准备模型及数据集**

```shell
# 下载MobileNet预测模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar -xf MobileNetV1_infer.tar
# 下载ImageNet小型数据集
wget https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz
tar -xf ILSVRC2012_data_demo.tar.gz
```

- **2.运行自动化压缩**

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

- **3.精度测试**

  - 测试压缩前模型的精度:

    ```shell
    CUDA_VISIBLE_DEVICES=0 python ./image_classification/eval.py
    ### Eval Top1: 0.7171724759615384
    ```

  - 测试量化模型的精度:

    ```shell
    CUDA_VISIBLE_DEVICES=0 python ./image_classification/eval.py --model_dir='MobileNetV1_quant'
    ### Eval Top1: 0.7166466346153846
    ```

  - 量化后模型的精度相比量化前的模型几乎精度无损，由于是使用的超参搜索的方法来选择的量化参数，所以每次运行得到的量化模型精度会有些许波动。

- **4.推理速度测试**

  - 量化模型速度的测试依赖推理库的支持，所以确保安装的是带有TensorRT的PaddlePaddle。以下示例和展示的测试结果是基于Tesla V100、CUDA 10.2、python3.7得到的。

  - 使用以下指令查看本地cuda版本，并且在[下载链接](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)中下载对应cuda版本和对应python版本的paddlepaddle安装包。

    ```shell
    cat /usr/local/cuda/version.txt ### CUDA Version 10.2.89
    ### 10.2.89 为cuda版本号，可以根据这个版本号选择需要安装的带有TensorRT的PaddlePaddle安装包。
    ```

  - 安装下载的whl包：（这里通过wget下载到的是python3.7、cuda10.2的PaddlePaddle安装包，若您的环境和示例环境不同，请依赖您自己机器的环境下载对应的安装包，否则运行示例代码会报错。）

    ```
    wget https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_x86_64.whl
    pip install paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_x86_64.whl --force-reinstall
    ```

  - 测试FP32模型的速度

    ```
    python ./image_classification/infer.py
    ### using tensorrt FP32    batch size: 1 time(ms): 0.6140608787536621
    ```

  - 测试FP16模型的速度

    ```
    python ./image_classification/infer.py --use_fp16=True
    ### using tensorrt FP16    batch size: 1 time(ms): 0.5795984268188477
    ```

  - 测试INT8模型的速度

    ```
    python ./image_classification/infer.py --model_dir=./MobileNetV1_quant/ --use_int8=True
    ### using tensorrt INT8 batch size: 1 time(ms): 0.5213963985443115
    ```

  - **提示：**

    - DataLoader传入的数据集是待压缩模型所用的数据集，DataLoader继承自`paddle.io.DataLoader`。可以直接使用模型套件中的DataLoader，或者根据[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)自定义所需要的DataLoader。
    - 自动化压缩Config中定义量化、蒸馏、剪枝等压缩算法会合并执行，压缩策略有：量化+蒸馏，剪枝+蒸馏等等。示例中选择的配置为离线量化超参搜索。
    - 如果要压缩的模型参数是存储在各自分离的文件中，需要先通过[convert.py](./convert.py) 脚本将其保存成一个单独的二进制文件。

## 进阶使用

- ACT可以自动处理常见的预测模型，如果有更特殊的改造需求，可以参考[ACT超参配置教程](./hyperparameter_tutorial.md)来进行单独配置压缩策略。

## 社区交流

- 微信扫描二维码并填写问卷之后，加入技术交流群

  <div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/178181077-57a3a631-f495-4821-878d-ef5e74981718.jpg"  width = "150" height = "150" />
  </div>

- 如果你发现任何关于ACT自动化压缩工具的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSlim/issues)给我们提issues。同时欢迎贡献更多优秀模型，共建开源生态。

## License

本项目遵循[Apache-2.0开源协议](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/LICENSE)

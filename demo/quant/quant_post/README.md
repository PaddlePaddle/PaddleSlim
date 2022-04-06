# 静态离线量化示例

本示例将介绍如何使用离线量化接口``paddleslim.quant.quant_post_static``来对训练好的分类模型进行离线量化, 无需对模型进行训练即可得到量化模型，减少模型的存储空间和显存占用。 本demo中模型均从[PaddleClas模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md) 中下载。

## 接口介绍

请参考 <a href='https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post-static'>量化API文档</a>。

## 分类模型的离线量化流程

### 环境准备

PaddlePaddle >= 2.3 或develop版本

### 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型
离线量化接口支持加载通过``paddle.static.save_inference_model``接口或者`paddle.jit.save`保存的静态图Inference模型。因此如果您的模型是通过其他接口保存的，需要先将模型进行转化。

图像分类的Inference模型均可从[PaddleClas模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)的表格中下载得到。

- MobileNetV1模型准备：
```
wget -P inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
cd inference_model/
tar -xf MobileNetV1_infer.tar
```

- ResNet50模型准备：
```
wget -P inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
cd inference_model/
tar -xf ResNet50_infer.tar
```

### 静态离线量化
接下来对导出的模型文件进行静态离线量化，静态离线量化的脚本为[quant_post.py](./quant_post.py)，脚本中使用接口``paddleslim.quant.quant_post_static``对模型进行离线量化。运行命令为：

```
# MobileNetV1
python quant_post.py --model_path ./inference_model/MobileNetV1_infer/ --save_path ./quant_model/MobileNet
# ResNet50
python quant_post.py --model_path ./inference_model/ResNet50_infer/ --save_path ./quant_model/ResNet50
```

- 参数列表：
| 参数名     | 解释 |
| :-------- | :--------: |
| model_path | 需要量化的模型所在路径 |
| save_path | 量化后的模型保存的路径 |
| model_filename | 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的模型文件名称，如果参数文件保存在多个文件中，则不需要设置。 |
| params_filename  | 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的参数文件名称，如果参数文件保存在多个文件中，则不需要设置。 |
| algo  | 激活量化使用的算法，默认是`hist`  |
| batch_size | 模型校准使用的batch size大小  |
| batch_num | 模型校准时的总batch数量 |
|  round_type  | 模型量化时四舍五入的方法，可选择`round`和`adaround`，默认是`round`   |
|  onnx_format | 保存量化模型时的格式是否是ONNX通配格式，默认False  |
|  input_name |  量化时模型输入的name，如果使用PaddleClas模型库中下载好的模型，保持默认为inputs，如果是自己导出模型，应设置：`--input_name='x'`，可用VisualDL或Netron查看模型输入正确name  |


运行以上命令后，可在``${save_path}``下看到量化后的模型文件和参数文件。

### 测试精度

使用[eval.py](./eval.py)脚本对量化前后的模型进行测试，得到模型的分类精度进行对比。

- 首先测试量化前的模型的精度，运行以下命令：
```shell
# MobileNetV1
python eval.py --model_path=./inference_model/MobileNetV1_infer --model_name=inference.pdmodel --params_name=inference.pdiparams
# ResNet50
python eval.py --model_path=./inference_model/ResNet50_infer --model_name=inference.pdmodel --params_name=inference.pdiparams
```

- 测试离线量化后的模型的精度：

```shell
# MobileNetV1
python eval.py --model_path ./quant_model/MobileNet/
# ResNet50
python eval.py --model_path ./quant_model/ResNet50/
```


### benchmark

| 模型     | FP32 acc-top1 | INT8 acc-top1  | INT8 acc(adaround) |
| :-------- | :--------: | :--------: |
| MobileNetV1 | 0.7092  | 0.7036  | 0.7063  |
| ResNet50 |  0.7633 | 0.7615 | 0.7625  |

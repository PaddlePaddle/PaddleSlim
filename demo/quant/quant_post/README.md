# 静态离线量化示例

本示例介绍如何使用离线量化接口``paddleslim.quant.quant_post_static``来对训练好的分类模型进行离线量化, 该接口无需对模型进行训练就可得到量化模型，减少模型的存储空间和显存占用。

## 接口介绍

请参考 <a href='https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post-static'>量化API文档</a>。

## 分类模型的离线量化流程

### 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型
因为离线量化接口只支持加载通过``paddle.static.save_inference_model``接口保存的模型，因此如果您的模型是通过其他接口保存的，那需要先将模型进行转化。本示例将以分类模型为例进行说明。

首先在[imagenet分类模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E5%B7%B2%E5%8F%91%E5%B8%83%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E6%80%A7%E8%83%BD)中下载训练好的``mobilenetv1``模型。

在当前文件夹下创建``'pretrain'``文件夹，将``mobilenetv1``模型在该文件夹下解压，解压后的目录为``pretrain/MobileNetV1_pretrained``

### 导出模型
通过运行以下命令可将模型转化为离线量化接口可用的模型：
```
python export_model.py --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --data imagenet
```
转化之后的模型存储在``inference_model/MobileNet/``文件夹下，可看到该文件夹下有``'model'``, ``'weights'``两个文件。

### 静态离线量化
接下来对导出的模型文件进行静态离线量化，静态离线量化的脚本为[quant_post.py](./quant_post.py)，脚本中使用接口``paddleslim.quant.quant_post_static``对模型进行离线量化。运行命令为：
```
python quant_post_static.py --model_path ./inference_model/MobileNet --save_path ./quant_model_train/MobileNet --model_filename model --params_filename weights
```

- ``model_path``: 需要量化的模型坐在的文件夹
- ``save_path``: 量化后的模型保存的路径
- ``model_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的模型文件名称，如果参数文件保存在多个文件中，则不需要设置。
- ``params_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的参数文件名称，如果参数文件保存在多个文件中，则不需要设置。

运行以上命令后，可在``${save_path}``下看到量化后的模型文件和参数文件。

> 使用的量化算法为``'KL'``, 使用训练集中的160张图片进行量化参数的校正。


### 测试精度

使用[eval.py](./eval.py)脚本对量化前后的模型进行测试，得到模型的分类精度进行对比。

首先测试量化前的模型的精度，运行以下命令：
```
python eval.py --model_path ./inference_model/MobileNet --model_name model --params_name weights
```
精度输出为:
```
top1_acc/top5_acc= [0.70913923 0.89548034]
```

使用以下命令测试离线量化后的模型的精度：

```
python eval.py --model_path ./quant_model_train/MobileNet --model_name __model__ --params_name __params__
```

精度输出为
```
top1_acc/top5_acc= [0.70141864 0.89086477]
```
从以上精度对比可以看出，对``mobilenet``在``imagenet``上的分类模型进行离线量化后 ``top1``精度损失为``0.77%``， ``top5``精度损失为``0.46%``.

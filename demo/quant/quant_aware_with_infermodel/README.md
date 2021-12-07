# 静态离线量化超参搜索示例

本示例将介绍如何使用infermodel进行蒸馏量化训练，使用接口``paddleslim.quant.quant_aware_with_infermodel``来进行量化蒸馏训练，
使用接口``paddleslim.quant.export_quant_infermodel``来进行量化蒸馏训练。

## 分类模型量化训练流程

### 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型
离线量化接口只支持加载通过``paddle.static.save_inference_model``接口保存的模型。因此如果您的模型是通过其他接口保存的，需要先将模型进行转化。本示例将以分类模型为例进行说明。

首先在[imagenet分类模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E5%B7%B2%E5%8F%91%E5%B8%83%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E6%80%A7%E8%83%BD)中下载训练好的``mobilenetv1``模型。

在当前文件夹下创建``'pretrain'``文件夹，将``mobilenetv1``模型在该文件夹下解压，解压后的目录为``pretrain/MobileNetV1_pretrained``

### 导出模型
通过运行以下命令可将模型转化为离线量化接口可用的模型：
```
python ../quant_post/export_model.py --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --data imagenet
```
转化之后的模型存储在``inference_model/MobileNet/``文件夹下，可看到该文件夹下有``'model'``, ``'weights'``两个文件。

### 量化蒸馏训练
接下来对导出的模型文件进行量化训练，脚本为[quant_aware_with_infermodel.py](./quant_aware_with_infermodel.py)，脚本中使用接口``paddleslim.quant.quant_aware_with_infermodel``对模型进行量化训练。运行命令为：
```
python quant_aware_with_infermodel.py \
    --use_gpu=True \
    --model_path="./inference_model/MobileNet/" \
    --checkpoint_path="./inference_model/MobileNet_checkpoints/" \
    --model_filename="model" \
    --params_filename="weights" \
```

- ``model_path``: 需要量化的模型所在路径
- ``checkpoint_path``: 量化训练模型checkpoint保存路径
- ``model_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的模型文件名称，如果参数文件保存在多个文件中，则不需要设置。
- ``params_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的参数文件名称，如果参数文件保存在多个文件中，则不需要设置。

运行以上命令后，可在``${checkpoint_path}``下看到量化后模型的checkpoint。

## 量化模型导出

把量化模型checkpoint导出为infermodel。

```
python export_quantmodel.py \
    --use_gpu=True \
    --checkpoint_path="./inference_model/MobileNet_checkpoints/epoch_0_iter_5" \
    --infermodel_save_path="./inference_model/MobileNet_epoch_0_iter_5_infermodel" \
```

### 测试精度

使用[eval.py](../quant_post/eval.py)脚本对量化前后的模型进行测试，得到模型的分类精度进行对比。

首先测试量化前的模型的精度，运行以下命令：
```
python ../quant_post/eval.py --model_path ./inference_model/MobileNet_epoch_0_iter_5_infermodel --model_name model --params_name params
```
精度输出为:
```
top1_acc/top5_acc= [0.70653 0.89369]
```

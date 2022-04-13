# 静态离线量化超参搜索示例

本示例将介绍如何使用离线量化超参搜索接口``paddleslim.quant.quant_post_hpo``来对训练好的分类模型进行离线量化超参搜索。

## 分类模型的离线量化超参搜索流程

### 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型
离线量化接口只支持加载预测模型，预测模型保存接口： 动态图使用paddle.jit.save保存； 静态图使用paddle.static.save_inference_model保存。因此如果您的模型是通过其他接口保存的，需要先将模型进行转化。本示例将以分类模型为例进行说明。

飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，该工具已提供分类预测模型下载链接。

MobileNetV1预测模型下载链接：https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar

```
mkdir inference_model
cd inference_model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar xf MobileNetV1_infer.tar
cd ..
```

### 静态离线量化
接下来对导出的模型文件进行静态离线量化，静态离线量化的脚本为[quant_post_hpo.py](./quant_post_hpo.py)，脚本中使用接口``paddleslim.quant.quant_post_hpo``对模型进行离线量化。运行命令为：
```
python quant_post_hpo.py \
    --use_gpu=True \
    --model_path="./inference_model/MobileNetV1_infer/" \
    --save_path="./inference_model/MobileNetV1_infer_quant/" \
    --model_filename="inference.pdmodel" \
    --params_filename="inference.pdiparams" \
    --max_model_quant_count=26
```

- ``model_path``: 需要量化的模型所在路径
- ``save_path``: 量化后的模型保存的路径
- ``model_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的模型文件名称，如果参数文件保存在多个文件中，则不需要设置。
- ``params_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的参数文件名称，如果参数文件保存在多个文件中，则不需要设置。
- ``max_model_quant_count``: 最大离线量化搜索次数，次数越多产出高精度量化模型概率越大，耗时也会相应增加。建议值：大于20小于30。

运行以上命令后，可在``${save_path}``下看到量化后的模型文件和参数文件。


### 测试精度

使用[eval.py](../quant_post/eval.py)脚本对量化前后的模型进行测试，得到模型的分类精度进行对比。

首先测试量化前的模型的精度，运行以下命令：
```
python ../quant_post/eval.py --model_path ./inference_model/MobileNetV1_infer/ --model_name inference.pdmodel --params_name inference.pdiparams
```
精度输出为:
```
top1_acc/top5_acc= [0.70903133 0.89540042]
```

使用以下命令测试离线量化后的模型的精度：

```
python ../quant_post/eval.py --model_path ./inference_model/MobileNet_quant/ --model_name __model__ --params_name __params__
```

精度输出为
```
top1_acc/top5_acc= [0.70890231, 0.89324836]
```

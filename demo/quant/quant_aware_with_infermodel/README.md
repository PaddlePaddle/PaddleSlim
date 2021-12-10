# 使用infermodel进行量化训练示例

本示例将介绍如何使用infermodel进行蒸馏量化训练，使用接口``paddleslim.quant.quant_aware_with_infermodel``训练量化模型，
使用接口``paddleslim.quant.export_quant_infermodel``导出训练好的量化模型，保存为infermodel形式。

## 分类模型量化训练流程

### 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型

本功能只支持加载通过``paddle.static.save_inference_model``接口保存的模型。因此如果您的模型是通过其他接口保存的，需要先将模型进行转化。本示例将以分类模型为例进行说明。

首先在[imagenet分类模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E5%B7%B2%E5%8F%91%E5%B8%83%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E6%80%A7%E8%83%BD)中下载训练好的``mobilenetv1``模型。

在当前文件夹下创建``'pretrain'``文件夹，将``mobilenetv1``模型在该文件夹下解压，解压后的目录为``pretrain/MobileNetV1_pretrained``

### 导出模型
通过运行以下命令可将模型转化为接口可用的模型：
```
python ../quant_post/export_model.py --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --data imagenet
```
转化之后的模型存储在``inference_model/MobileNet/``文件夹下，可看到该文件夹下有``'model'``, ``'weights'``两个文件。

### 量化蒸馏训练
接下来对导出的模型文件进行量化训练，脚本为[quant_aware_with_infermodel.py](./quant_aware_with_infermodel.py)，脚本中使用接口``paddleslim.quant.quant_aware_with_infermodel``对模型进行量化训练。运行命令为：
```
python quant_aware_with_infermodel.py \
    --use_gpu=False \
    --batch_size=2 \
    --num_epoch=30 \
    --save_iter_step=100 \
    --learning_rate=0.0001 \
    --weight_decay=0.00004 \
    --use_pact=False \
    --checkpoint_path="./inference_model/MobileNet_quantaware_ckpt/" \
    --model_path="./inference_model/MobileNet/" \
    --model_filename="model" \
    --params_filename="weights" \
    --teacher_model_path="./inference_model/MobileNet/" \
    --teacher_model_filename="model" \
    --teacher_params_filename="weights" \
    --distill_node_name_list "teacher_fc_0.tmp_0" "fc_0.tmp_0" "teacher_batch_norm_24.tmp_4" "batch_norm_24.tmp_4" \
            "teacher_batch_norm_22.tmp_4" "batch_norm_22.tmp_4" "teacher_batch_norm_18.tmp_4" "batch_norm_18.tmp_4" \
            "teacher_batch_norm_13.tmp_4" "batch_norm_13.tmp_4" "teacher_batch_norm_5.tmp_4" "batch_norm_5.tmp_4"
```

- ``batch_size``: 量化训练batch size
- ``num_epoch``: 量化训练epoch数
- ``save_iter_step``: 每隔save_iter_step保存一次checkpoint
- ``learning_rate``: 量化训练学习率，推荐使用float模型训练最小一级学习率
- ``weight_decay``: 推荐使用float模型训练weight decay设置
- ``use_pact``: 是否使用pact量化算法
- ``checkpoint_path``: 量化训练模型checkpoint保存路径
- ``model_path``: 需要量化的模型所在路径
- ``model_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的模型文件名称，如果参数文件保存在多个文件中，则不需要设置。
- ``params_filename``: 如果需要量化的模型的参数文件保存在一个文件中，则设置为该模型的参数文件名称，如果参数文件保存在多个文件中，则不需要设置。
- ``teacher_model_path``: teacher模型所在路径, 可以和量化模型是同一个，即自蒸馏
- ``teacher_model_filename``: teacher模型model文件名字
- ``teacher_params_filename``: teacher模型参数文件名字
- ``distill_node_name_list``: 蒸馏节点名字列表，每两个节点组成一对，分别属于teacher模型和量化模型

运行以上命令后，可在``${checkpoint_path}``下看到量化后模型的checkpoint。

## 量化模型导出

把量化模型checkpoint导出为infermodel。

```
python export_quantmodel.py \
    --use_gpu=False \
    --batch_size=128 \
    --num_epoch=30 \
    --save_iter_step=3000 \
    --learning_rate=0.0001 \
    --weight_decay=0.00004 \
    --use_pact=False \
    --checkpoint_path="./inference_model/MobileNet_quantaware_ckpt/" \
    --model_path="./inference_model/MobileNet/" \
    --model_filename="model" \
    --params_filename="weights" \
    --teacher_model_path="./inference_model/MobileNet/" \
    --teacher_model_filename="model" \
    --teacher_params_filename="weights" \
    --distill_node_name_list "teacher_fc_0.tmp_0" "fc_0.tmp_0" "teacher_batch_norm_24.tmp_4" "batch_norm_24.tmp_4" \
            "teacher_batch_norm_22.tmp_4" "batch_norm_22.tmp_4" "teacher_batch_norm_18.tmp_4" "batch_norm_18.tmp_4" \
            "teacher_batch_norm_13.tmp_4" "batch_norm_13.tmp_4" "teacher_batch_norm_5.tmp_4" "batch_norm_5.tmp_4"
    --checkpoint_path="./inference_model/MobileNet_checkpoints/epoch_0_iter_5" \
    --infermodel_save_path="./inference_model/MobileNet_epoch_0_iter_5_infermodel" \
```

- ``checkpoint_filename``: 需要导出的量化模型checkpoint文件名，如epoch_0_iter_6000
- ``infermodel_save_path``: 导出infermodel保存路径

### 测试精度

使用[eval.py](../quant_post/eval.py)脚本对量化前后的模型进行测试，得到模型的分类精度进行对比。

首先测试量化前的模型的精度，运行以下命令：
```
python eval.py --model_path ./inference_model/MobileNet --model_name model --params_name weights
```
精度输出为:
```
top1_acc/top5_acc= [0.70653 0.89369]
```

使用以下命令测试离线量化训练后的模型的精度：
```
python ../quant_post/eval.py --model_path ./inference_model/MobileNet_epoch_0_iter_6000_infermodel --model_name model --params_name params
```
精度输出为:
```
top1_acc/top5_acc= [0.70334 0.89374]]
```

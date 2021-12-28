# 使用预测模型进行量化训练示例

预测模型获取
动态图使用paddle.jit.save保存；
静态图使用paddle.static.save_inference_model保存。

本示例将介绍如何使用预测模型进行蒸馏量化训练，
首先使用接口``paddleslim.quant.quant_aware_with_infermodel``训练量化模型，
训练完成后，使用接口``paddleslim.quant.export_quant_infermodel``将训好的量化模型导出为预测模型。

## 分类模型量化训练流程

###1. 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 2. 准备需要量化的模型

飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，本示例使用该套件产出imagenet分类模型。
####① 下载MobileNetV2预训练模型
预训练模型库地址 ``https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md``
MobileNetV2预训练模型地址 ``https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams``
在PaddleClas代码库根目录创建pretrained文件夹，MobileNetV2预训练参数保存在该文件夹中。

#### ② 导出预测模型
PaddleClas代码库根目录执行如下命令，导出预测模型
```
python tools/export_model.py \
    -c ppcls/configs/ImageNet/MobileNetV2/MobileNetV2.yaml \
    -o Global.pretrained_model=pretrained/MobileNetV2_pretrained \
    -o Global.save_inference_dir=infermodel_mobilenetv2 \
```
#### ③ 测试模型精度
使用[eval.py](../quant_post/eval.py)脚本得到模型的分类精度：
```
python ../quant_post/eval.py --model_path infermodel_mobilenetv2 --model_name inference.pdmodel --params_name inference.pdiparams
```
精度输出为:
```
top1_acc/top5_acc= [0.71918 0.90568]
```

### 3. 进行量化蒸馏训练

蒸馏量化训练示例脚本为[quant_aware_with_infermodel.py](./quant_aware_with_infermodel.py)，使用接口``paddleslim.quant.quant_aware_with_infermodel``对模型进行量化训练。运行命令为：
```
python quant_aware_with_infermodel.py \
    --use_gpu=False \
    --batch_size=2 \
    --num_epoch=30 \
    --save_iter_step=20 \
    --learning_rate=0.0001 \
    --weight_decay=0.00004 \
    --use_pact=False \
    --checkpoint_path="./inference_model/MobileNet_quantaware_ckpt/" \
    --model_path_prefix="./inference_model/infermodel_mobilenetv2/inference" \
    --teacher_model_path_prefix="./inference_model/infermodel_mobilenetv2/inference" \
    --distill_node_name_list "teacher_conv2d_54.tmp_0" "conv2d_54.tmp_0" "teacher_conv2d_55.tmp_0" "conv2d_55.tmp_0" \
        "teacher_conv2d_57.tmp_0" "conv2d_57.tmp_0" "teacher_elementwise_add_0" "elementwise_add_0" \
        "teacher_conv2d_61.tmp_0" "conv2d_61.tmp_0" "teacher_elementwise_add_1" "elementwise_add_1" \
        "teacher_elementwise_add_2" "elementwise_add_2" "teacher_conv2d_67.tmp_0" "conv2d_67.tmp_0" \
        "teacher_elementwise_add_3" "elementwise_add_3" "teacher_elementwise_add_4" "elementwise_add_4" \
        "teacher_elementwise_add_5" "elementwise_add_5" "teacher_conv2d_75.tmp_0" "conv2d_75.tmp_0" \
        "teacher_elementwise_add_6" "elementwise_add_6" "teacher_elementwise_add_7" "elementwise_add_7" \
        "teacher_conv2d_81.tmp_0" "conv2d_81.tmp_0" "teacher_elementwise_add_8" "elementwise_add_8" \
        "teacher_elementwise_add_9" "elementwise_add_9" "teacher_conv2d_87.tmp_0" "conv2d_87.tmp_0" \
        "teacher_linear_1.tmp_0" "linear_1.tmp_0"
```
- ``batch_size``: 量化训练batch size。
- ``num_epoch``: 量化训练epoch数。
- ``save_iter_step``: 每隔save_iter_step保存一次checkpoint。
- ``learning_rate``: 量化训练学习率，推荐使用float模型训练最小一级学习率。
- ``weight_decay``: 推荐使用float模型训练weight decay设置。
- ``use_pact``: 是否使用pact量化算法, 推荐使用。
- ``checkpoint_path``: 量化训练模型checkpoint保存路径。
- ``model_path_prefix``: 需要量化模型的目录 + 模型名称（不包含后缀）
- ``teacher_model_path_prefix``: teacher模型的目录 + 模型名称（不包含后缀）, 可以和量化模型是同一个，即自蒸馏。
- ``distill_node_name_list``: 蒸馏节点名字列表，每两个节点组成一对，分别属于teacher模型和量化模型。

运行以上命令后，可在``${checkpoint_path}``下看到量化后模型的checkpoint。

### 4. 量化模型导出

量化模型checkpoint导出为预测模型。

```
python export_quantmodel.py \
    --use_gpu=False \
    --batch_size=2 \
    --num_epoch=30 \
    --save_iter_step=20 \
    --learning_rate=0.0001 \
    --weight_decay=0.00004 \
    --use_pact=False \
    --checkpoint_path="./inference_model/MobileNet_quantaware_ckpt/" \
    --model_path_prefix="./inference_model/infermodel_mobilenetv2/inference" \
    --teacher_model_path_prefix="./inference_model/infermodel_mobilenetv2/inference" \
    --distill_node_name_list "teacher_conv2d_54.tmp_0" "conv2d_54.tmp_0" "teacher_conv2d_55.tmp_0" "conv2d_55.tmp_0" \
        "teacher_conv2d_57.tmp_0" "conv2d_57.tmp_0" "teacher_elementwise_add_0" "elementwise_add_0" \
        "teacher_conv2d_61.tmp_0" "conv2d_61.tmp_0" "teacher_elementwise_add_1" "elementwise_add_1" \
        "teacher_elementwise_add_2" "elementwise_add_2" "teacher_conv2d_67.tmp_0" "conv2d_67.tmp_0" \
        "teacher_elementwise_add_3" "elementwise_add_3" "teacher_elementwise_add_4" "elementwise_add_4" \
        "teacher_elementwise_add_5" "elementwise_add_5" "teacher_conv2d_75.tmp_0" "conv2d_75.tmp_0" \
        "teacher_elementwise_add_6" "elementwise_add_6" "teacher_elementwise_add_7" "elementwise_add_7" \
        "teacher_conv2d_81.tmp_0" "conv2d_81.tmp_0" "teacher_elementwise_add_8" "elementwise_add_8" \
        "teacher_elementwise_add_9" "elementwise_add_9" "teacher_conv2d_87.tmp_0" "conv2d_87.tmp_0" \
        "teacher_linear_1.tmp_0" "linear_1.tmp_0" \
    --checkpoint_filename="epoch_0_iter_20" \
    --export_inference_model_path_prefix="./inference_model/MobileNet_quantaware_ckpt/epoch_0_iter_20_infer"
```
- ``checkpoint_filename``: checkpoint文件名。
- ``export_inference_model_path_prefix``: 量化模型导出的目录 + 模型名称（不包含后缀）。

###5. 测试精度

使用[eval.py](../quant_post/eval.py)脚本对量化后的模型进行精度测试：
```
python ../quant_post/eval.py --model_path ./quant_infermodel_mobilenetv2/ --model_name model --params_name params
```
精度输出为:
```
top1_acc/top5_acc= [0.71764 0.90418]
```

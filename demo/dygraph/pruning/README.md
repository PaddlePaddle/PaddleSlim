# 剪裁示例

该示例介绍了如何使用PaddleSlim对PaddlePaddle动态图进行剪裁。
使用的数据集有ImageNet1K与Cifar10，支持的模型有：Mobilenet系列分类模型、Resnet系列模型。

## 1. 数据准备

### 1.1 ImageNet1K

数据下载链接：http://www.image-net.org/challenges/LSVRC/2012/

下载数据后，按以下结构组织数据：

```
PaddleClas/dataset/ILSVRC2012/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt
```

上述结构中的 `train_list.txt` 与 `val_list.txt` 内容如下：

```
# delimiter: "space"
# content of train_list.txt
train/n01440764/n01440764_10026.JPEG 0
...

# content of val_list.txt
val/ILSVRC2012_val_00000001.JPEG 65
...
```

### 1.2 Cifar10

对于`Cifar10`数据，该示例直接使用的是`paddle.vision.dataset.Cifar10`提供的数据读取接口，该接口会自动下载数据并将其缓存到本地文件系统，用户不需要关系该数据集的存储与格式。

## 2. 剪裁与训练

实践表明，对在目标任务上预训练过的模型进行剪裁，比剪裁没训过的模型，最终的效果要好。该示例中直接使用`paddle.vision.models`模块提供的针对`ImageNet1K`分类任务的预训练模型。
对预训练好的模型剪裁后，需要在目标数据集上进行重新训练，以便恢复因剪裁损失的精度。
在`train.py`脚本中实现了上述剪裁和重训练两个步骤，其中的可配置参数可以通过执行`python train.py --help`查看。


### 2.1 CPU训练或GPU单卡训练

执行如下命令在GPU单卡进行剪裁和训练，该参数列表表示：对在`ImageNet1K`数据集上预训练好的`resnet34`模型进行剪裁，每层卷积剪掉25%的`filters`，卷积内评估`filters`重要性的方式为`FPGM`。最后对训练好的模型重训练120个epoch，并将每个epoch产出的模型保存至`./fpgm_resnet34_025_120_models`路径下。

```
python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=120 \
    --batch_size=256 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models"
```

如果需要仅在CPU上训练，需要修改上述命令中的`--use_gpu`为`False`.

### 2.2 GPU多卡训练

以下命令为启动GPU多卡剪裁和重训练任务，任务内容与2.1节内容一致。其中需要注意的是：`batch_size`为多张卡上总的`batch_size`。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
--gpus="0,1,2,3" \
--log_dir="fpgm_resnet34_f-42_train_log" \
train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --batch_size=256 \
    --num_epochs=120 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models"
```

### 2.3 恢复训练

通过设置`checkpoint`选项进行恢复训练：

```
python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=120 \
    --batch_size=256 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" \
    --checkpoint="./fpgm_resnet34_025_120_models/0"
```

## 3. 评估

通过调用`eval.py`脚本，对剪裁和重训练后的模型在测试数据上进行精度：


```
python eval.py \
--checkpoint=./fpgm_resnet34_025_120_models/1 \
--model="resnet34" \
--pruned_ratio=0.25
```

## 4. 导出模型

执行以下命令导出用于预测的模型：

```
python export_model.py \
--checkpoint=./fpgm_resnet34_025_120_models/1 \
--model="resnet34" \
--pruned_ratio=0.25 \
--output_path=./infer/resnet
```

如上述命令所示，如果指定了`--output_path=./infer/resnet`，则会在路径`./infer`下生成三个文件：`resnet.pdiparams`, `resnet.pdmodel`, `resnet.pdiparams.info`. 这三个文件可以被PaddleLite或PaddleInference加载使用。

## 5. 部分实验结果

| 模型        | 原模型精度（Top1/Top5）     | FLOPs剪裁百分比  | 剪裁后模型准确率(Top1/Top5) | 使用脚本                       |
| ----------- | --------------------------- | ---------------- | --------------------------- | ------------------------------ |
| MobileNetV1 | 70.99/89.68                 | -50%             | 69.23/88.71                 | fpgm_mobilenetv1_f-50_train.sh |
| MobileNetV2 | 72.15/90.65                 | -50%             | 67.00/87.56                 | fpgm_mobilenetv2_f-50_train.sh |
| ResNet34    | 74.57/92.14                 | -42%             | 73.20/91.21                 | fpgm_resnet34_f-42_train.sh    |

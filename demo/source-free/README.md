# 使用预测模型进行量化训练示例

预测模型保存接口：
动态图使用``paddle.jit.save``保存；
静态图使用``paddle.static.save_inference_model``保存。

本示例将介绍如何使用预测模型进行蒸馏量化训练，
首先使用接口``paddleslim.quant.quant_aware_with_infermodel``训练量化模型，
训练完成后，使用接口``paddleslim.quant.export_quant_infermodel``将训好的量化模型导出为预测模型。

## 分类模型量化训练流程

### 1. 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 2. 准备需要量化的模型

飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，本示例使用该套件产出imagenet分类模型。
#### 2.1 下载PaddleClas release/2.3分支代码
<https://github.com/PaddlePaddle/PaddleClas/archive/refs/heads/release/2.3.zip>
解压后，进入PaddleClas目录
```
cd PaddleClas-release-2.3
```
#### 2.2 下载MobileNetV2预训练模型
在PaddleClas根目录创建``pretrained``文件夹：
```
mkdir pretrained
```

下载预训练模型
分类预训练模型库地址 <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md>
MobileNetV2预训练模型地址 <https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams>
执行下载命令：
```
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams -O ./pretrained/MobileNetV2_pretrained.pdparams
```

#### 2.3 导出预测模型
PaddleClas代码库根目录执行如下命令，导出预测模型
```
python tools/export_model.py \
    -c ppcls/configs/ImageNet/MobileNetV2/MobileNetV2.yaml \
    -o Global.pretrained_model=pretrained/MobileNetV2_pretrained \
    -o Global.save_inference_dir=infermodel_mobilenetv2
```
#### 2.4 测试模型精度
拷贝``infermodel_mobilenetv2``文件夹到``PaddleSlim/demo/source-free/``文件夹。
```
cd PaddleSlim/demo/source-free/
```
使用[eval.py](../quant_post/eval.py)脚本得到模型的分类精度：
```
python ../quant_post/eval.py --model_path infermodel_mobilenetv2 --model_name inference.pdmodel --params_name inference.pdiparams
```
精度输出为:
```
top1_acc/top5_acc= [0.71918 0.90568]
```

### 3. 进行多策略融合压缩

每一个小章节代表一种多策略融合压缩，不代表需要串行执行。

### 3.1 进行量化蒸馏压缩
蒸馏量化训练示例脚本为[qat_demo.py](./qat_demo.py)，使用接口``paddleslim.source_free.AutoCompression``对模型进行量化训练。运行命令为：
```
python qat_demo.py \
    --model_dir='infermodel_mobilenetv2' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_qat_mbv2/' \
    --devices='gpu' \
    --batch_size=64 \
    --distill_loss='l2_loss' \
    --distill_node_pair "teacher_conv2d_54.tmp_0" "conv2d_54.tmp_0" "teacher_conv2d_55.tmp_0" "conv2d_55.tmp_0" \
            "teacher_conv2d_57.tmp_0" "conv2d_57.tmp_0" "teacher_elementwise_add_0" "elementwise_add_0" \
            "teacher_conv2d_61.tmp_0" "conv2d_61.tmp_0" "teacher_elementwise_add_1" "elementwise_add_1" \
            "teacher_elementwise_add_2" "elementwise_add_2" "teacher_conv2d_67.tmp_0" "conv2d_67.tmp_0" \
            "teacher_elementwise_add_3" "elementwise_add_3" "teacher_elementwise_add_4" "elementwise_add_4" \
            "teacher_elementwise_add_5" "elementwise_add_5" "teacher_conv2d_75.tmp_0" "conv2d_75.tmp_0" \
            "teacher_elementwise_add_6" "elementwise_add_6" "teacher_elementwise_add_7" "elementwise_add_7" \
            "teacher_conv2d_81.tmp_0" "conv2d_81.tmp_0" "teacher_elementwise_add_8" "elementwise_add_8" \
            "teacher_elementwise_add_9" "elementwise_add_9" "teacher_conv2d_87.tmp_0" "conv2d_87.tmp_0" \
            "teacher_linear_1.tmp_0" "linear_1.tmp_0" \
    --distill_lambda=1.0 \
    --teacher_model_dir='./infermodel_mobilenetv2' \
    --teacher_model_filename='inference.pdmodel' \
    --teacher_params_filename='inference.pdiparams' \
    --epochs=1 \
    --optimizer='SGD' \
    --learning_rate=0.0001 \
    --weight_decay=0.00004 \
    --eval_iter=1000 \
    --origin_metric=0.71918
```

### 3.2 进行离线量化超参搜索压缩
离线量化超参搜索压缩示例脚本为[ptq_demo.py](./ptq_demo.py)，使用接口``paddleslim.source_free.AutoCompression``对模型进行压缩。运行命令为：
```
python ptq_demo.py \
    --model_dir='infermodel_mobilenetv2' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_qat_mbv2/' \
    --devices='gpu' \
    --batch_size=64 \
    --ptq_algo "KL" "hist" \
    --bias_correct True False \
    --weight_quantize_type "channel_wise_abs_max" "abs_max" \
    --hist_percent 0.9999 0.99999 \
    --batch_num 4 16 \
    --max_quant_count 20
```

### 3.3 进行剪枝蒸馏策略融合压缩
注意：ASP 暂时只支持对 FC 进行剪枝, 所以示例为对BERT模型进行ASP稀疏。
首先参考[脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#%E9%A2%84%E6%B5%8B)得到可部署的模型。
剪枝蒸馏压缩示例脚本为[asp_demo.py](./asp_demo.py)，使用接口``paddleslim.source_free.AutoCompression``对模型进行压缩。运行命令为：
```
python asp_demo.py \
    --model_dir='./static_bert_models/' \
    --model_filename='bert.pdmodel' \
    --params_filename='bert.pdiparams' \
    --save_dir='./save_asp_bert/' \
    --devices='gpu' \
    --batch_size=32 \
    --prune_algo='asp' \
    --distill_loss='l2_loss' \
    --distill_node_pair "teacher_tmp_9" "tmp_9" "teacher_tmp_12" "tmp_12"\
            "teacher_tmp_15" "tmp_15" "teacher_tmp_18" "tmp_18" \
            "teacher_tmp_21" "tmp_21" "teacher_tmp_24" "tmp_24" \
            "teacher_tmp_27" "tmp_27" "teacher_tmp_30" "tmp_30" \
            "teacher_tmp_33" "tmp_33" "teacher_tmp_36" "tmp_36" \
            "teacher_tmp_39" "tmp_39" "teacher_tmp_42" "tmp_42" \
            "teacher_linear_147.tmp_1" "linear_147.tmp_1" \
    --distill_lambda=1.0 \
    --teacher_model_dir='./static_bert_models/' \
    --teacher_model_filename='bert.pdmodel' \
    --teacher_params_filename='bert.pdiparams' \
    --epochs=3 \
    --optimizer='AdamW' \
    --learning_rate=2e-5 \
    --eval_iter=1000 \
    --origin_metric=0.93 \
```

### 3.4 进行非结构化稀疏蒸馏策略融合压缩
非结构化稀疏蒸馏压缩示例脚本为[unstructure_prune_demo.py](./unstructure_prune_demo.py)，使用接口``paddleslim.source_free.AutoCompression``对模型进行压缩。运行命令为：
```
python unstructure_prune_demo.py
```

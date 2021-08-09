1. 准备

安装需要测试的Paddle版本和PaddleSlim版本。

准备ImageNet数据集，数据集需要满足paddle hapi的要求。假定解压到`/dataset/ILSVRC2012`文件夹，该文件夹下有`train文件夹、val文件夹、train_list.txt和val_list.txt文件`。

通过`export CUDA_VISIBLE_DEVICES=xx`指定需要使用的GPU ID。

2. 产出PTQ量化模型

在`run_ptq.sh`中设置数据路径，比如`data_path="/dataset/ILSVRC2012"`。

执行`sh run_ptq.sh`，会使用动态图离线量化方法量化mobilenetv1、mobilenetv2、resnet50和vgg16模型。

执行完成，量化模型保存在当前`output_ptq`目录下。
比如`output_ptq/mobilenet_v1/fp32_infer`是原始FP32模型，`output_ptq/mobilenet_v1/int8_infer`是PTQ量化模型。

3. 产出QAT量化模型

在`run_qat.sh`文件中设置`data_path`为上述ImageNet数据集的路径`/dataset/ILSVRC2012`。

执行`sh run_train.sh` 会对几个分类模型使用动态图量化训练功能进行量化，其中只执行一个epoch。

执行完后，在`output_qat`目录下有产出的量化模型。
比如`output_qat/mobilenet_v1`是QAT量化模型。

4. 转换量化模型

在X86 CPU上部署量化模型，需要使用`src/save_quant_model.py`脚本对量化模型进行转换。

如下是对`output_qat/mobilenet_v1`模型进行转换的示例。
```
sh run_convert.sh output_qat/mobilenet_v1 int8_qat_models/mobilenet_v1
```

按照上述示例，将所有QAT和PTQ产出的量化模型进行转换，假定分别保存在`int8_qat_models`和`int8_ptq_models`文件中。

4. 测试模型

在`run_test.sh`脚本中设置`data_path`为上述ImageNet数据集的路径`/dataset/ILSVRC2012`。

使用`run_test.sh`脚本测试原始FP32模型（共4个）的精度，可以开启GPU，举例如下。
```
sh run_test.sh output_ptq/mobilenet_v1/fp32_infer/mobilenet_v1 True
```

使用`run_test.sh`脚本测试转换前PTQ和QAT量化模型(分别4个)的精度，可以开启GPU，举例如下。
```
sh run_test.sh output_qat/mobilenet_v1 True
```

使用`run_test.sh`脚本测试转换后PTQ和QAT量化模型（分别4个）的精度，不可以开启GPU，举例如下。
```
sh run_test.sh int8_qat_models/mobilenet_v1 False
```

5. 测试目标

使用动态图量化训练功能，产出`mobilenet_v1`,`mobilenet_v2`,`resnet50`,`vgg16`量化模型，测试转换前后量化模型精度在1%误差范围内。

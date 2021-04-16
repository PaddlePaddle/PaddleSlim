1. 准备

安装需要测试的Paddle版本和PaddleSlim版本。

准备ImageNet数据集，假定解压到`/dataset/ILSVRC2012`文件夹，该文件夹下有`train文件夹、val文件夹、train_list.txt和val_list.txt文件`。

2. 产出量化模型

在`run_train.sh`文件中设置`data_path`为上述ImageNet数据集的路径`/dataset/ILSVRC2012`。

根据实际情况，在`run_train.sh`文件中设置使用GPU的id等参数。

执行`sh run_train.sh` 会对几个分类模型使用动态图量化训练功能进行量化，其中只执行一个epoch。
执行完后，在`output_models/quant_dygraph`目录下有产出的量化模型。

3. 转换量化模型

在Intel CPU上部署量化模型，需要使用`test/save_quant_model.py`脚本进行模型转换。

如下是对`mobilenet_v1`模型进行转换的示例。
```
python src/save_quant_model.py --load_model_path output_models/quant_dygraph/mobilenet_v1 --save_model_path int8_models/mobilenet_v1
```

4. 测试量化模型

在`run_test.sh`脚本中设置`data_path`为上述ImageNet数据集的路径`/dataset/ILSVRC2012`。

根据实际情况，在`run_test.sh`文件中设置使用GPU的id等参数。

使用`run_test.sh`脚本测试转换前和转换后的量化模型精度。

比如：
```
sh run_test.sh output_models/quant_dygraph/mobilenet_v1
sh run_test.sh int8_models/mobilenet_v1
```

5. 测试目标

使用动态图量化训练功能，产出`mobilenet_v1`,`mobilenet_v2`,`resnet50`,`vgg16`量化模型，测试转换前后量化模型精度在1%误差范围内。



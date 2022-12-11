# QAT(Quantization-Aware Training)量化分析工具详细教程

## 1. 量化分析工具功能
精度误差分析(metric_error_analyse)：
 - 遍历量化训练后模型的每层，去掉量化节点并计算当前层不量化的模型精度。该功能可以定位具体某层导致的量化损失。


## 2. paddleslim.quant.AnalysisQAT 可传入参数解析
| **参数名**                   | **参数释义**                              |
|-----------------------------|-----------------------------------------|
| quant_model_dir | 必须传入的量化后的模型文件路径 |
| float_model_dir | 必须传入的量化前的模型文件路径 |
| model_filename | 默认为None，若model_dir为文件夹名，则必须传入以'.pdmodel'结尾的模型名称 |
| params_filename | 默认为None，若model_dir为文件夹名，则必须传入以'.pdiparams'结尾的模型名称 |
| quantizable_op_type | 需分析的量化的op类型，默认为`conv2d`, `depthwise_conv2d`, `mul` |
| qat_metric | 量化模型的精度，可不传入，默认为None，不传入时会自动计算 |
| eval_function | 若需要验证精度，需要传入自定义的验证函数；若不传入，精度误差分析将根据Cosine Similarity计算得出 |
| data_loader | 模型校准时使用的数据，DataLoader继承自`paddle.io.DataLoader`。可以直接使用模型套件中的DataLoader，或者根据[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)自定义所需要的DataLoader |
| save_dir | 分析后保存模型精度或pdf等文件的文件夹，默认为`analysis_results`|
| resume | 是否加载中间分析文件，默认为False|





## 3. 量化分析工具的使用
**创建量化分析工具** ：
```shell
# 下载Inference模型
wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar -xf MobileNetV1_infer.tar
wget -q https://paddle-slim-models.bj.bcebos.com/act/MobileNetV1_QAT.tar
tar -xf MobileNetV1_QAT.tar

# 下载demo数据集
wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz
tar -xf ILSVRC2012_data_demo.tar.gz
```

```shell

import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddleslim.quant.analysis_qat import AnalysisQAT

paddle.enable_static()


class ImageNetDataset(DatasetFolder):
    def __init__(self, path, image_size=224):
        super(ImageNetDataset, self).__init__(path)
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(image_size),
            transforms.Transpose(), normalize
        ])

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.samples)

train_dataset = ImageNetDataset(
    "./ILSVRC2012_data_demo/ILSVRC2012/train/")
image = paddle.static.data(
    name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
train_loader = paddle.io.DataLoader(
    train_dataset, feed_list=[image], batch_size=8, return_list=False)

analyzer = AnalysisQAT(
    float_model_dir="./MobileNetV1_infer",
    quant_model_dir="./MobileNetV1_QAT",
    model_filename="inference.pdmodel",
    params_filename="inference.pdiparams",
    save_dir="MobileNetV1_analysis",
    data_loader=train_loader)
```


**精度误差分析**
```shell
analyzer.metric_error_analyse()
```
调用该接口，会遍历量化模型中的每一层，去掉量化节点并计算当前层不量化的模型精度。调用该接口时，需要输入Eval Function。会产出所有去掉一层量化的模型精度排序，将默认保存在 `./analysis_results/analysis.txt` 中。具体使用可参考[GPT量化训练敏感度分析DEMO](../../../../example/quantization_analysis/GPT/README.md)。


## FAQ：
- 与PTQ(Post Training Quantization)量化分析工具的区别：与PTQ量化分析工具不同的是，QAT量化分析工具加载量化训练后的量化模型，遍历所有量化的层，依次去掉量化层，加载Float模型的参数，并进行验证获取精度误差分析。而PTQ量化分析工具则是加载待量化的原模型，对模型所有层依次进行量化，每次量化一层，进行验证获取精度误差分析。

- QAT量化分析工具设计的原因：QAT量化分析工具依次去掉量化层，而不是依次量化一层是由于QAT需要训练的特性。遍历每层进行量化训练再验证精度比较耗时，直接加载量化训练后的量化模型，依次去掉量化层更高效。

- 量化分析工具为什么要区分PTQ和QAT：实验证明PTQ和QAT后的量化模型的敏感层并不完全一致，将两种算法分开，敏感度分析结果更加准确。

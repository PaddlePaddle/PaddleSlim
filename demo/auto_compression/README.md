# 自动化压缩工具ACT（Auto Compression Toolkit）

## 简介
PaddleSlim推出全新自动化压缩工具（ACT），旨在通过Source-Free的方式，自动对预测模型进行压缩，压缩后模型可直接部署应用。ACT自动化压缩工具主要特性如下：
- **『更便捷』**：开发者无需了解或修改模型源码，直接使用导出的预测模型进行压缩；
- **『更智能』**：开发者简单配置即可启动压缩，ACT工具会自动优化得到最好预测模型；
- **『更丰富』**：ACT中提供了量化训练、蒸馏、结构化剪枝、非结构化剪枝、多种离线量化方法及超参搜索等等，可任意搭配使用。


## 环境准备

- 安装PaddlePaddle >= 2.3版本 （从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- 安装PaddleSlim develop版本

## 快速上手

- 1.准备模型及数据集

```shell
# 下载MobileNet预测模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar -xf MobileNetV1_infer.tar
# 下载ImageNet小型数据集
wget https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz
tar xf ILSVRC2012_data_demo.tar.gz
```

- 2.运行

```python
# 导入依赖包
import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddleslim.auto_compression import AutoCompression
paddle.enable_static()
# 定义DataSet
class ImageNetDataset(DatasetFolder):
    def __init__(self, path, image_size=224):
        super(ImageNetDataset, self).__init__(path)
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size), transforms.Transpose(),
            normalize
        ])

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.samples)

# 定义DataLoader
train_dataset = ImageNetDataset("./ILSVRC2012_data_demo/ILSVRC2012/train/")
image = paddle.static.data(
    name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
train_loader = paddle.io.DataLoader(train_dataset, feed_list=[image], batch_size=32, return_list=False)
# 开始自动压缩
ac = AutoCompression(
    model_dir="./MobileNetV1_infer",
    model_filename="inference.pdmodel",
    params_filename="inference.pdiparams",
    save_dir="output",
    strategy_config=None,
    train_config=None,
    train_dataloader=train_loader,
    eval_dataloader=train_loader)  # eval_function to verify accuracy
ac.compress()
```

**提示：**
- DataLoader传入的数据集是待压缩模型所用的数据集，DataLoader继承自`paddle.io.DataLoader`。
- 如无需验证自动化压缩过程中模型的精度，`eval_callback`可不传入function，程序会自动根据损失来选择最优模型。
- 自动化压缩Config中定义量化、蒸馏、剪枝等压缩算法会合并执行，压缩策略有：量化+蒸馏，剪枝+蒸馏等等。

## 应用示例

#### [图像分类](./image_classification)

#### [目标检测](./detection)

#### [语义分割](./semantic_segmentation)

#### [NLP](./nlp)

#### 即将发布
- [ ] 更多自动化压缩应用示例
- [ ] X2Paddle模型自动化压缩示例

## 其他

- ACT可以自动处理常见的预测模型，如果有更特殊的改造需求，可以参考[ACT超参配置教程](./hyperparameter_tutorial.md)来进行单独配置压缩策略。

- 如果你发现任何关于ACT自动化压缩工具的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSlim/issues)给我们提issues。同时欢迎贡献更多优秀模型，共建开源生态。

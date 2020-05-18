#  图像分类模型通道剪裁-敏感度分析

该教程以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的敏感度分析接口](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#sensitivity)。
该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 定义输入数据
4. 定义模型评估方法
5. 训练模型
6. 获取待分析卷积参数名称
7. 分析敏感度
8. 剪裁模型

以下章节依次次介绍每个步骤的内容。

## 1. 导入依赖

PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
```

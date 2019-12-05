
# 卷积通道剪裁API文档

## class Pruner

---

>paddleslim.prune.Pruner(criterion="l1_norm")[源代码]()

对卷积网络的通道进行一次剪裁。剪裁一个卷积层的通道，是指剪裁该卷积层输出的通道。卷积层的权重形状为`[output_channel, input_channel, kernel_size, kernel_size]`，通过剪裁该权重的第一纬度达到剪裁输出通道数的目的。

**参数：**

    - **criterion:** 评估一个卷积层内通道重要性所参考的指标。目前仅支持`l1_norm`。默认为`l1_norm`。

**返回：** 一个Pruner类的实例

**示例代码：**

```
from paddleslim.prune import Pruner
pruner = Pruner()
```

---

>prune(program, scope, params, ratios, place=None, lazy=False, only_graph=False, param_backup=None, param_shape_backup=None)

对目标网络的一组卷积层的权重进行裁剪。

**参数：**

- **program(paddle.fluid.Program):** 要裁剪的目标网络。更多关于Program的介绍请参考：[Program概念介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program)。

- **scope(paddle.fluid.Scope):** 要裁剪的权重所在的`scope`，Paddle中用`scope`实例存放模型参数和运行时变量的值。更多介绍请参考[Scope概念介绍]()

- **params(list<str>):** 需要被裁剪的卷积层的参数的名称列表。可以通过以下方式查看模型中所有参数的名称:
```
for block in program.blocks:
    for param in block.all_parameters():
        print("param: {}; shape: {}".format(param.name, param.shape))
```

- **ratios(list<float>):** 用于裁剪`params`的剪切率，类型为列表。该列表长度必须与`params`的长度一致。

- **place(paddle.fluid.Place):** 

---

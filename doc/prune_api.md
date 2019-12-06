
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

- **scope(paddle.fluid.Scope):** 要裁剪的权重所在的`scope`，Paddle中用`scope`实例存放模型参数和运行时变量的值。Scope中的参数值会被`inplace`的裁剪。更多介绍请参考[Scope概念介绍]()

- **params(list<str>):** 需要被裁剪的卷积层的参数的名称列表。可以通过以下方式查看模型中所有参数的名称:
```
for block in program.blocks:
    for param in block.all_parameters():
        print("param: {}; shape: {}".format(param.name, param.shape))
```

- **ratios(list<float>):** 用于裁剪`params`的剪切率，类型为列表。该列表长度必须与`params`的长度一致。

- **place(paddle.fluid.Place):** 待裁剪参数所在的设备位置，可以是`CUDAPlace`或`CPUPLace`。[Place概念介绍]()

- **lazy(bool):** `lazy`为True时，通过将指定通道的参数置零达到裁剪的目的，参数的`shape保持不变`；`lazy`为False时，直接将要裁的通道的参数删除，参数的`shape`会发生变化。

- **only_graph(bool):** 是否只裁剪网络结构。在Paddle中，Program定义了网络结构，Scope存储参数的数值。一个Scope实例可以被多个Program使用，比如定义了训练网络的Program和定义了测试网络的Program是使用同一个Scope实例的。`only_graph`为True时，只对Program中定义的卷积的通道进行剪裁；`only_graph`为false时，Scope中卷积参数的数值也会被剪裁。默认为False。

- **param_backup(bool):** 是否返回对参数值的备份。默认为False。

- **param_shape_backup(bool):** 是否返回对参数`shape`的备份。

**返回：**

- **pruned_program(paddle.fluid.Program):** 被裁剪后的Program。

- **param_backup(dict):** 对参数数值的备份，用于恢复Scope中的参数数值。

- **param_shape_backup(dict):** 对参数形状的备份。

**示例：**

```

import paddle.fluid as fluid
from paddleslim.prune import Pruner

def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  name,
                  stride=1,
                  groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + "_out")
    bn_name = name + "_bn"
    return fluid.layers.batch_norm(
        input=conv,
        act=act,
        name=bn_name + '_output',
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance', )

main_program = fluid.Program()
startup_program = fluid.Program()
#   X       X              O       X              O
# conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
#     |            ^ |                    ^
#     |____________| |____________________|
#
# X: prune output channels
# O: prune input channels
with fluid.program_guard(main_program, startup_program):
    input = fluid.data(name="image", shape=[None, 3, 16, 16])
    conv1 = conv_bn_layer(input, 8, 3, "conv1")
    conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
    sum1 = conv1 + conv2
    conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
    conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
    sum2 = conv4 + sum1
    conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
    conv6 = conv_bn_layer(conv5, 8, 3, "conv6")

place = fluid.CPUPlace()
exe = fluid.Executor(place)
scope = fluid.Scope()
exe.run(startup_program, scope=scope)
pruner = Pruner()
main_program, _, _ = pruner.prune(
    main_program,
    scope,
    params=["conv4_weights"],
    ratios=[0.5],
    place=place,
    lazy=False,
    only_graph=False,
    param_backup=None,
    param_shape_backup=None)

for param in main_program.global_block().all_parameters():
    if "weights" in param.name:
        print("param name: {}; param shape: {}".format(param.name, param.shape))

```


---

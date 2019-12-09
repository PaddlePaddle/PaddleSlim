# 模型分析API文档

## flops

>paddleslim.analysis.flops(program, detail=False) [源代码]()

获得指定网络的每秒浮点运算次数(FLOPS)。

**参数：**

- **program(paddle.fluid.Program):**  待分析的目标网络。更多关于Program的介绍请参考：[Program概念介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program)。

- **detail(bool):** 是否返回每个卷积层的FLOPS。默认为False。

**返回值：**

- **flops(float):** 整个网络的FLOPS。

- **params2flops(dict):** 每层卷积对应的FLOPS，其中key为卷积层参数名称，value为FLOPS值。

**示例：**

```
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.analysis import flops

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

print("FLOPS: {}".format(flops(main_program)))
```

## model_size

>paddleslim.analysis.model_size(program) [源代码]()

获得指定网络的参数数量。

**参数：**

- **program(paddle.fluid.Program):**  待分析的目标网络。更多关于Program的介绍请参考：[Program概念介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program)。

**返回值：**

- **model_size(int):** 整个网络的参数数量。

**示例：**

```
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.analysis import model_size

def conv_layer(input,
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
    return conv

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
    conv1 = conv_layer(input, 8, 3, "conv1")
    conv2 = conv_layer(conv1, 8, 3, "conv2")
    sum1 = conv1 + conv2
    conv3 = conv_layer(sum1, 8, 3, "conv3")
    conv4 = conv_layer(conv3, 8, 3, "conv4")
    sum2 = conv4 + sum1
    conv5 = conv_layer(sum2, 8, 3, "conv5")
    conv6 = conv_layer(conv5, 8, 3, "conv6")

print("FLOPS: {}".format(model_size(main_program)))
```

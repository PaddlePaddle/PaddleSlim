
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

点击[AIStudio](https://aistudio.baidu.com/aistudio/projectDetail/200786)执行以下示例代码。
```

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
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

## sensitivity

>paddleslim.prune.sensitivity(program, place, param_names, eval_func, sensitivities_file=None, pruned_ratios=None) [源代码]()

计算网络中每个卷积层的敏感度。每个卷积层的敏感度信息统计方法为：依次剪掉当前卷积层不同比例的输出通道数，在测试集上计算剪裁后的精度损失。得到敏感度信息后，可以通过观察或其它方式确定每层卷积的剪裁率。

**参数：**

- **program(paddle.fluid.Program):** 待评估的目标网络。更多关于Program的介绍请参考：[Program概念介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program)。

- **place(paddle.fluid.Place):** 待分析的参数所在的设备位置，可以是`CUDAPlace`或`CPUPLace`。[Place概念介绍]()

- **param_names(list<str>):** 待分析的卷积层的参数的名称列表。可以通过以下方式查看模型中所有参数的名称:

```
for block in program.blocks:
    for param in block.all_parameters():
        print("param: {}; shape: {}".format(param.name, param.shape))
```

- **eval_func(function):** 用于评估裁剪后模型效果的回调函数。该回调函数接受被裁剪后的`program`为参数，返回一个表示当前program的精度，用以计算当前裁剪带来的精度损失。

- **sensitivities_file(str):** 保存敏感度信息的本地文件系统的文件。在敏感度计算过程中，会持续将新计算出的敏感度信息追加到该文件中。重启任务后，文件中已有敏感度信息不会被重复计算。该文件可以用`pickle`加载。

- **pruned_ratios(list<float>):** 计算卷积层敏感度信息时，依次剪掉的通道数比例。默认为[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]。

**返回：**

- **sensitivities(dict):** 存放敏感度信息的dict，其格式为：

```
{"weight_0": 
   {"loss": [0.22, 0.33],
    "pruned_percent": [0.1, 0.2]
   },
 "weight_1":
   {"loss": [0.21, 0.4],
    "pruned_percent": [0.1, 0.2]
   }
}
```

其中，`weight_0`是卷积层参数的名称，`weight_0`对应的`loss[i]`为将`weight_0`裁掉`pruned_percent[i]`后的精度损失。

**示例：**

点击[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/201401)运行以下示例代码。

```
import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.prune import sensitivity
import paddle.dataset.mnist as reader

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
image_shape = [1,28,28]
with fluid.program_guard(main_program, startup_program):
    image = fluid.data(name='image', shape=[None]+image_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')   
    conv1 = conv_bn_layer(image, 8, 3, "conv1")
    conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
    sum1 = conv1 + conv2
    conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
    conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
    sum2 = conv4 + sum1
    conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
    conv6 = conv_bn_layer(conv5, 8, 3, "conv6")
    out = fluid.layers.fc(conv6, size=10, act="softmax")
#    cost = fluid.layers.cross_entropy(input=out, label=label)
#    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
#    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
 
   
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)

val_reader = paddle.batch(reader.test(), batch_size=128)
val_feeder = feeder = fluid.DataFeeder(
        [image, label], place, program=main_program)

def eval_func(program):

    acc_top1_ns = []
    for data in val_reader():
        acc_top1_n = exe.run(program,
                             feed=val_feeder.feed(data),
                             fetch_list=[acc_top1.name])
        acc_top1_ns.append(np.mean(acc_top1_n))
    return np.mean(acc_top1_ns)
param_names = []
for param in main_program.global_block().all_parameters():
    if "weights" in param.name:
        param_names.append(param.name)
sensitivities = sensitivity(main_program,
                            place,
                            param_names,
                            eval_func,
                            sensitivities_file="./sensitive.data",
                            pruned_ratios=[0.1, 0.2, 0.3])
print(sensitivities)

```

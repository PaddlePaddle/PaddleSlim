# One Shot NAS 示例

>该示例依赖Paddle1.7.0或Paddle develop版本。

该示例使用MNIST数据，介绍了如何使用PaddleSlim的OneShotNAS接口搜索出一个分类网络。OneShotNAS仅支持动态图，所以该示例完全使用Paddle动态图模式。

## 关键代码介绍

One-shot网络结构搜索策略包含以下步骤：

1. 定义超网络
2. 训练超网络
3. 基于超网络搜索子网络
4. 训练最佳子网络

以下按序介绍各个步骤的关键代码。

### 定义超级网络

按照动态图教程，定义一个分类网络模块，该模块包含4个子模块：`_simple_img_conv_pool_1`,`_simple_img_conv_pool_2`,`super_net`和`fc`，其中`super_net`为`SuperMnasnet`的一个实例。

在前向计算过程中，输入图像先后经过子模块`_simple_img_conv_pool_1`、`super_net`、`_simple_img_conv_pool_2`和`fc`的前向计算。

代码如下所示：
```
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConv(1, 20, 2, act="relu")
        self.arch = SuperMnasnet(
            name_scope="super_net", input_channels=20, out_channels=20)
        self._simple_img_conv_pool_2 = SimpleImgConv(20, 50, 2, act="relu")

        self.pool_2_shape = 50 * 13 * 13
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    def forward(self, inputs, label=None, tokens=None):
        x = self._simple_img_conv_pool_1(inputs)

        x = self.arch(x, tokens=tokens)  # addddddd
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

```

动态图模块MNIST的forward函数接受一个参数`tokens`，用于指定在前向计算中使用的子网络，如果`tokens`为None，则随机选取一个子网络进行前向计算。

### 训练超级网络

网络训练的逻辑定义在`train_mnist`函数中，将`tokens`参数设置为None，进行超网络训练，即在每个batch选取一个超网络进行训练。

代码如下所示：

```
with fluid.dygraph.guard(place):
    model = MNIST()
    train_mnist(args, model)
```

### 搜索最佳子网络
使用PaddleSlim提供的`OneShotSearch`接口搜索最佳子网络。传入已定义且训练好的超网络实例`model`和一个用于评估子网络的回调函数`test_mnist`.

代码如下：

```
best_tokens = OneShotSearch(model, test_mnist)
```

### 训练最佳子网络

获得最佳的子网络的编码`best_tokens`后，调用之前定义的`train_mnist`方法进行子网络的训练。代码如下：

```
train_mnist(args, model, best_tokens)
```

## 启动示例

执行以下代码运行示例：

```
python train.py
```

执行`python train.py --help`查看更多可配置选项。

## FAQ

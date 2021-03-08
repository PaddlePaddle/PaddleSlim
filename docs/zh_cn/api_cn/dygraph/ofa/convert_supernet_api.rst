Convert SuperNet
============

在进行Once-For-All训练之前，需要把普通的模型先转换为由动态OP组网的超网络。超网络转换在把普通网络转换为超网络的同时也会把超网络中的最大的子网络转换为搜索空间中最大的网络。

.. note::
  - 如果原始卷积的kernel_size是1，则不会对它的kernel_size进行改变。
..

接口介绍
------------------

.. py:class:: paddleslim.nas.ofa.supernet(kernel_size=None, expand_ratio=None, channel=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/ofa/convert_super.py#L643>`_

通过键值对的方式传入搜索空间。

**参数：**
  - **kernel_size(list|tuple, optional)：** 网络中Conv2D的kernel_size的搜索空间。
  - **expand_ratio(list|tuple, optional)：** 网络中Conv2D的通道数、Embedding和Linear的参数输出维度的搜索空间，本参数是按照原始模型中每个OP的通道的比例来得到转换后的超网络中每个OP的通道数，所以本参数的长度为1。本参数和 ``channel`` 之间设置一个即可。
  - **channel(list(list)|tuple(tuple), optional)：** 网络中Conv2D的通道数、Embedding和Linear的参数输出维度的搜索空间，本参数是直接设置超网络中每个OP的通道数量，所以本参数的长度需要和网络中包括的Conv2D、Embedding、Linear的总数相等。本参数和 ``expand_ratio`` 之间设置一个即可。

**返回：**
超网络配置。

.. py:class:: paddleslim.nas.ofa.Convert(context)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/ofa/convert_super.py#L45>`_

把普通网络根据传入的自定义的搜索空间转换为超网络。

**返回：**
转换实例

**参数：**
  - **context(paddleslim.nas.ofa.supernet)：** 用户自定义的搜索空间

  .. py:method:: convert(network)

  实际超网络转换。

  **参数：**
    - **network(paddle.nn.Layer)：** 要转换为超网络的原始模型实例。

  **返回：**
  实例化之后的超网络。

PaddleSlim提供了三种方式构造超网络，下面分别介绍这三种方式。

方式一
------------------
直接调用搜索空间定义接口和超网络转换接口转换超网络。这种方式的优点是不需要重新定义网络，直接对初始化之后的网络实例进行转换，缺点是只能对整个网络进行超网络转换，不能对部分网络进行超网络转换。

**示例代码：**

.. code-block:: python

  from paddle.vision.models import mobilenet_v1
  from paddleslim.nas.ofa.convert_super import Convert, supernet

  model = mobilenet_v1()
  sp_net_config = supernet(kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4])
  sp_model = Convert(sp_net_config).convert(self.model)

方式二
------------------
使用上下文的方式转换超网络。这种方式的优点是可以仅转换部分网络为超网络，或者对网络不同部分进行不同的超网络转换，缺点是需要拿到原始网络的定义，并修改网络定义。

**示例代码：**

.. code-block:: python

  import paddle.nn as nn
  from paddleslim.nas.ofa.convert_super import supernet

  class Net(nn.Layer):
    def __init__(self):
      super(Net, self).__init__()
      models = []
      with supernet(kernel_size=(3, 5, 7), expand_ratio=(1, 2, 4)) as ofa_super:
        models += [nn.Conv2D(3, 4, 3, padding=1)]
        models += [nn.InstanceNorm2D(4)]
        models = ofa_super.convert(models)
      models += [nn.Conv2D(4, 4, 3, groups=4)]
      self.models = paddle.nn.Sequential(*models)

     def forward(self, inputs):
       return self.models(inputs)

方式三
------------------
直接调用动态OP组网，组网方式和普通模型相同。PaddleSlim支持的动态OP请参考 `动态OP <./ofa_layer_api.rst>`_ 。这种方式的优点是组网更自由，缺点是用法更复杂。

.. note::
  - paddleslim.nas.ofa.layers 文件中的动态OP是基于Paddle 2.0beta及其之后的版本实现的。paddleslim.nas.ofa.layers_old文件中的动态OP是基于Paddle 2.0beta之前的版本实现的。
  - Block接口是把当前动态OP的搜索空间加入到OFA训练过程中的搜索空间中。由于Conv2D、Embedding、Linear这三个OP的参数中输出的部分是可以随意修改的，所以这三个OP所对应的动态OP需要使用Block包装一下。而Norm相关的动态OP由于其参数大小是根据输入大小相关，所以不需要用Block包装。
..

**示例代码：**

.. code-block:: python

  import paddle.nn as nn
  from paddleslim.nas.ofa.layers import Block, SuperConv2D, SuperBatchNorm2D

  class Net(nn.Layer):
    def __init__(self):
      super(Net, self).__init__()
      self.models = [Block(SuperConv2D(3, 4, 3, candidate_config={'kernel_size': (3, 5, 7), 'channel': (4, 8, 16)}))]
      self.models += [SuperBatchNorm2D(16)]

    def forward(self, inputs):
        return self.models(inputs)

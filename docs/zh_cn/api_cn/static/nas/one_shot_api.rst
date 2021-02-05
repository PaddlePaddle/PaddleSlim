OneShotNAS
=============

OneShotSearch
------------------

.. py:function:: paddleslim.nas.one_shot.OneShotSearch(model, eval_func, strategy='sa', search_steps=100)

从超级网络中搜索出一个最佳的子网络。

**参数：**

- **model(fluid.dygraph.layer):** 通过在 ``OneShotSuperNet`` 前后添加若该模块构建的动态图模块。因为 ``OneShotSuperNet`` 是一个超网络，所以 ``model`` 也是一个超网络。换句话说，在 ``model`` 模块的子模块中，至少有一个是 ``OneShotSuperNet`` 的实例。该方法从 ``model`` 超网络中搜索得到一个最佳的子网络。超网络 ``model`` 需要先被训练，具体细节请参考[OneShotSuperNet]()。

- **eval_func:** 用于评估子网络性能的回调函数。该回调函数需要接受 ``model`` 为参数，并调用 ``model`` 的 ``forward`` 方法进行性能评估。

- **strategy(str):** 搜索策略的名称。默认为 ``sa`` ， 当前仅支持 ``sa`` .

- **search_steps(int):** 搜索轮次数。默认为100。

**返回：**

- **best_tokens:** 表示最佳子网络的编码信息（tokens）。

**示例代码：**

请参考[one-shot NAS示例]()


OneShotSuperNet
-----------------

.. py:class:: paddleslim.nas.one_shot.OneShotSuperNet(name_scope)

用于`OneShot`搜索策略的超级网络的基类，所有超级网络的实现要继承该类。

**参数：**

- **name_scope:(str) **超级网络的命名空间。

**返回：**

- **super_net:** 一个`OneShotSuperNet`实例。


   .. py:method:: init_tokens()
   
   获得当前超级网络的初始化子网络的编码，主要用于搜索。
   
   **返回：**
   
   - **tokens(list<int>):** 一个子网络的编码。
   
   range_table()
   
   : 超级网络中各个子网络由一组整型数字编码表示，该方法返回编码每个位置的取值范围。
   
   **返回：**
   
   - **range_table(tuple):** 子网络编码每一位的取值范围。 ``range_table`` 格式为 ``(min_values, max_values)`` ，其中， ``min_values`` 为一个整型数组，表示每个编码位置可选取的最小值； ``max_values`` 表示每个编码位置可选取的最大值。
   
   .. py:method:: _forward_impl(input, tokens)
   
   前向计算函数。 ``OneShotSuperNet`` 的子类需要实现该函数。
   
   **参数：**
   
   - **input(Variable):** 超级网络的输入。
   
   - **tokens(list<int>):** 执行前向计算所用的子网络的编码。默认为 ``None`` ，即随机选取一个子网络执行前向。
   
   **返回：**
   
   - **output(Variable):** 前向计算的输出
   
   .. py:method:: forward(self, input, tokens=None)
   
   执行前向计算。
   
   **参数：**
   
   - **input(Variable):** 超级网络的输入。
   
   - **tokens(list<int>):** 执行前向计算所用的子网络的编码。默认为 ``None`` ，即随机选取一个子网络执行前向。
   
   **返回：**
   
   - **output(Variable):** 前向计算的输出
   
   
   .. py:method:: _random_tokens()
   
   随机选取一个子网络，并返回其编码。
   
   **返回：**
   
   - **tokens(list<int>):** 一个子网络的编码。

SuperMnasnet
--------------


.. py:class:: paddleslim.nas.one_shot.SuperMnasnet(name_scope, input_channels=3, out_channels=1280, repeat_times=[6, 6, 6, 6, 6, 6], stride=[1, 1, 1, 1, 2, 1], channels=[16, 24, 40, 80, 96, 192, 320], use_auxhead=False)

在 `Mnasnet <https://arxiv.org/abs/1807.11626>`_ 基础上修改得到的超级网络, 该类继承自 ``OneShotSuperNet`` .

**参数：**

- **name_scope(str):** 命名空间。

- **input_channels(str):** 当前超级网络的输入的特征图的通道数量。

- **out_channels(str):** 当前超级网络的输出的特征图的通道数量。

- **repeat_times(list):** 每种 ``block`` 重复的次数。

- **stride(list):** 一种 ``block`` 重复堆叠成 ``repeat_block`` ， ``stride`` 表示每个 ``repeat_block`` 的下采样比例。

- **channels(list):** ``channels[i]`` 和 ``channels[i+1]`` 分别表示第i个 ``repeat_block`` 的输入特征图的通道数和输出特征图的通道数。

- **use_auxhead(bool):** 是否使用辅助特征图。如果设置为 ``True`` ，则 ``SuperMnasnet`` 除了返回输出特征图，还还返回辅助特征图。默认为False.

**返回：**

- **instance(SuperMnasnet):** 一个 ``SuperMnasnet`` 实例

**示例：**
.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   class MNIST(fluid.dygraph.Layer):
       def __init__(self):
           super(MNIST, self).__init__()
           self.arch = SuperMnasnet(
               name_scope="super_net", input_channels=20, out_channels=20)
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
   
           x = self.arch(inputs, tokens=tokens)
           x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
           x = self._fc(x)
           if label is not None:
               acc = fluid.layers.accuracy(input=x, label=label)
               return x, acc
           else:
               return x

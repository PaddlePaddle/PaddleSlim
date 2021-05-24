OFA-超网络转换
==============

PaddleSlim提供了一些API的动态版本，动态API指的是这些OP的参数大小可以在实际运行过程中根据传入的参数进行改变，用法上的差别具体是forward时候需要额外传一些实际运行相关的参数。其中 `layers_old.py <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers_old.py>`_ 对应的是Paddle 2.0alpha及之前版本的API， `layers.py <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py>`_ 对应的是Paddle 2.0alpha之后版本的API。

.. py:class:: paddleslim.nas.ofa.layers.Block(fn, fixed=False, key=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L64>`_

对Layer进行封装，封装后的Layer和普通Layer用法相同。把每层定义的搜索空间整合到一个大的搜索空间中，训练的时候可以去选择每层的搜索空间。只有在实际运行过程中可以主动改变参数大小的API需要用本类封装，即只有 ``Conv2D`` 、 ``Linear`` 和 ``Embedding`` 这三个API构造的层可能需要被封装。

**参数：**
  - **fn(paddle.nn.Layer)：** 需要被封装的层的实例。
  - **fixed(bool, optional)：** 在OFA训练过程中，本层的参数形状否保持不变，如果设置为False，则正常搜索，如果设置为True，则在OFA训练过程中本API的参数形状保持不变。默认：False。
  - **key(string, optional)：** 本层在整个搜索空间中对应的名称，默认：None。

**返回：**
Block实例

**示例代码：**

.. code-block:: python

  from paddleslim.nas.ofa.layers import Block, SuperConv2D
  
  block_layer = Block(SuperConv2D(3, 4, 3, candidate_config={'kerne_size': (3, 5, 7)}))

.. py:class:: paddleslim.nas.ofa.layers.SuperConv2D(in_channels, out_channels, kernel_size, candidate_config={}, transform_kernel=False, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format='NCHW')

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L85>`_

该接口用于构建 SuperConv2D 类的一个可调用对象。

**参数：**
  - **in_channels** (int) - 输入图像的通道数。
  - **out_channels** (int) - 由卷积操作产生的输出的通道数。
  - **kernel_size** (int) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
  - **candidate_config** （dict，可选）- 针对本层卷积的搜索空间，以字典的形式传入，字典可选的关键字包括： ``kernel_size`` ， ``expand_ratio``， ``channel`` ，其中 ``expand_ratio`` 和 ``channel`` 含义相同，都是对通道数进行搜索，不能同时设置。默认值：{}。
  - **transform_kernel** （bool，可选）- 是否使用转换矩阵把大kernel转换为小kernel。默认值：False。
  - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
  - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有3种格式：(1)包含4个二元组：当 ``data_format`` 为"NCHW"时为 [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NHWC"时为[[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；(2)包含4个整数值：[padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；(3)包含2个整数值：[padding_height, padding_width]，此时padding_height_top = padding_height_bottom = padding_height， padding_width_left = padding_width_right = padding_width。若为一个整数，padding_height = padding_width = padding。默认值：0。
  - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
  - **groups** (int，可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和卷积核分别根据通道数量平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值：1。
  - **padding_mode** (str, 可选): - 填充模式。 包括 ``'zeros'``, ``'reflect'``, ``'replicate'`` 或者 ``'circular'``. 默认值: ``'zeros'`` .
  - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

  .. py:method:: forward(input, kernel_size=None, expand_ratio=None, channel=None)

  **参数：**
    - **input** (Tensor)：- 实际输入。
    - **kernel_size** （int, 可选）：- 实际运行过程中卷积核大小，设置为None时则初始卷积核大小。默认：None。
    - **expand_ratio** （int|float, 可选）：- 实际运行过程中卷积核输出通道数膨胀比例，设置为None时则初始卷积核通道数。本参数和 ``channel`` 不能同时不为None。默认：None。
    - **channel** （int, 可选）：- 实际运行过程中卷积核输出通道数，设置为None时则初始卷积核通道数。本参数和 ``expand_ratio`` 不能同时不为None。默认：None。

**示例代码：**

.. code-block:: python

   import paddle 
   from paddleslim.nas.ofa.layers import SuperConv2D
   import numpy as np
   data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
   super_conv2d = SuperConv2D(3, 10, 3)
   config = {'channel': 5}
   data = paddle.to_tensor(data)
   conv = super_conv2d(data, **config)

.. py:class:: paddleslim.nas.ofa.layers.SuperConv2DTranspose(in_channels, out_channels, kernel_size, candidate_config={}, transform_kernel=False, stride=1, padding=0, output_padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format='NCHW')

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L386>`_

该接口用于构建 SuperConv2DTranspose 类的一个可调用对象。

**参数：**
  - **in_channels** (int) - 输入图像的通道数。
  - **out_channels** (int) - 卷积核的个数，和输出特征图通道数相同。
  - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
  - **candidate_config** （dict，可选）- 针对本层转置卷积的搜索空间，以字典的形式传入，字典可选的关键字包括： ``kernel_size`` ， ``expand_ratio``， ``channel`` ，其中 ``expand_ratio`` 和 ``channel`` 含义相同，都是对通道数进行搜索，不能同时设置。默认值：{}。
  - **transform_kernel** （bool，可选）- 是否使用转换矩阵把大kernel转换为小kernel。默认值：False。
  - **stride** (int|tuple, 可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含两个整型数，分别表示垂直和水平滑动步长。否则，表示垂直和水平滑动步长均为 ``stride`` 。默认值：1。
  - **padding** (int|tuple, 可选) - 填充大小。如果 ``padding`` 为元组或列表，则必须包含两个整型数，分别表示竖直和水平边界填充大小。否则，表示竖直和水平边界填充大小均为 ``padding`` 。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下方形状 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。默认值：0。
  - **output_padding** (int|list|tuple, optional): 输出形状上一侧额外添加的大小. 默认值: 0.
  - **groups** (int, 可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的分组卷积：当group=2，卷积核的前一半仅和输入特征图的前一半连接。卷积核的后一半仅和输入特征图的后一半连接。默认值：1。
  - **dilation** (int|tuple, 可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
  - **weight_attr** (ParamAttr, 可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool, 可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

  .. py:method:: forward(input, kernel_size=None, expand_ratio=None, channel=None)

  **参数：**
    - **input** (Tensor)：- 实际输入。
    - **kernel_size** （int, 可选）：- 实际运行过程中卷积核大小，设置为None时则初始卷积核大小。默认：None。
    - **expand_ratio** （int|float, 可选）：- 实际运行过程中卷积核输出通道数膨胀比例，设置为None时则初始卷积核通道数。本参数和 ``channel`` 不能同时不为None。默认：None。
    - **channel** （int, 可选）：- 实际运行过程中卷积核输出通道数，设置为None时则初始卷积核通道数。本参数和 ``expand_ratio`` 不能同时不为None。默认：None。

**示例代码：**

.. code-block:: python

  import paddle 
  from paddleslim.nas.ofa.layers import SuperConv2DTranspose
  import numpy as np
  data = np.random.uniform(-1, 1, [32, 10, 32, 32]).astype('float32')
  config = {'channel': 5}
  data = paddle.to_tensor(data)
  super_convtranspose = SuperConv2DTranspose(32, 10, 3)
  ret = super_convtranspose(paddle.to_tensor(data), **config)


.. py:class:: paddleslim.nas.ofa.layers.SuperLinear(in_features, out_features, candidate_config={}, weight_attr=None, bias_attr=None, name=None):

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L833>`_

该接口用于构建 SuperLinear 类的一个可调用对象。

**参数：**
  - **in_features** (int) – 线性变换层输入单元的数目。
  - **out_features** (int) – 线性变换层输出单元的数目。
  - **candidate_config** （dict，可选）- 针对本层Linear的搜索空间，以字典的形式传入，字典可选的关键字包括： ``expand_ratio``， ``channel`` ，其中 ``expand_ratio`` 和 ``channel`` 含义相同，都是对通道数进行搜索，不能同时设置。默认值：{}。
  - **weight_attr** (ParamAttr, 可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr, 可选) – 指定偏置参数属性的对象，若 `bias_attr` 为bool类型，如果设置为False，表示不会为该层添加偏置；如果设置为True，表示使用默认的偏置参数属性。默认值为None，表示使用默认的偏置参数属性。默认的偏置参数属性将偏置参数的初始值设为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **name** (string, 可选) – BatchNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

  .. py:method:: forward(input, expand_ratio=None, channel=None)

  **参数：**
    - **input** (Tensor)：- 实际输入。
    - **expand_ratio** （int|float, 可选）：- 实际运行过程中卷积核输出通道数膨胀比例，设置为None时则初始卷积核通道数。本参数和 ``channel`` 不能同时不为None。默认：None。
    - **channel** （int, 可选）：- 实际运行过程中卷积核输出通道数，设置为None时则初始卷积核通道数。本参数和 ``expand_ratio`` 不能同时不为None。默认：None。

**示例代码：**

.. code-block:: python

  import numpy as np
  import paddle
  from paddleslim.nas.ofa.layers import SuperLinear

  data = np.random.uniform(-1, 1, [32, 64]).astype('float32')
  config = {'channel': 16}
  linear = SuperLinear(64, 64)
  data = paddle.to_tensor(data)
  res = linear(data, **config)


.. py:class:: paddleslim.nas.ofa.layers.SuperEmbedding(num_embeddings, embedding_dim, candidate_config={}, padding_idx=None, sparse=False, weight_attr=None, name=None):

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L1131>`_

该接口用于构建 SuperEmbedding 类的一个可调用对象。

**参数：**
  - **num_embeddings** (int) - Embedding字典词表大小。
  - **embedding_dim** (int) - Embedding矩阵每个词向量的维度。
  - **candidate_config** （dict，可选）- 针对本层Embedding的搜索空间，以字典的形式传入，字典可选的关键字包括： ``expand_ratio``， ``channel`` ，其中 ``expand_ratio`` 和 ``channel`` 含义相同，都是对通道数进行搜索，不能同时设置。默认值：{}。
  - **padding_idx** (int|long|None) - padding_idx需在区间[-vocab_size, vocab_size)，否则不生效，padding_idx<0时，padding_idx会被改成vocab_size + padding_idx，input中等于padding_index的id对应的embedding信息会被设置为0，且这部分填充数据在训练时将不会被更新。如果为None，不作处理，默认为None。
  - **sparse** (bool) - 是否使用稀疏的更新方式，这个参数只会影响反向的梯度更新的性能，sparse更新速度更快，推荐使用稀疏更新的方式。但某些optimizer不支持sparse更新，比如 :ref:`cn_api_fluid_optimizer_AdadeltaOptimizer` 、 :ref:`cn_api_fluid_optimizer_AdamaxOptimizer` 、 :ref:`cn_api_fluid_optimizer_DecayedAdagradOptimizer` 、 :ref:`cn_api_fluid_optimizer_FtrlOptimizer` 、 :ref:`cn_api_fluid_optimizer_LambOptimizer` 、:ref:`cn_api_fluid_optimizer_LarsMomentumOptimizer` ，此时sparse必须为False。默认为False。
  - **weight_attr** (ParamAttr) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。此外，可以通过 ``weight_attr`` 参数加载用户自定义或预训练的词向量。只需将本地词向量转为numpy数据格式，且保证本地词向量的shape和embedding的 ``num_embeddings`` 和 ``embedding_dim`` 参数一致，然后使用 :ref:`cn_api_fluid_initializer_NumpyArrayInitializer` 进行初始化，即可实现加载自定义或预训练的词向量。详细使用方法见代码示例2。
  - **name** (string, 可选) – BatchNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

  .. py:method:: forward(input, kernel_size=None, expand_ratio=None, channel=None)

  **参数：**
    - **input** (Tensor)：- 实际输入。
    - **expand_ratio** （int|float, 可选）：- 实际运行过程中卷积核输出通道数膨胀比例，设置为None时则初始卷积核通道数。本参数和 ``channel`` 不能同时不为None。默认：None。
    - **channel** （int, 可选）：- 实际运行过程中卷积核输出通道数，设置为None时则初始卷积核通道数。本参数和 ``expand_ratio`` 不能同时不为None。默认：None。

**示例代码：**

.. code-block:: python

  import numpy as np
  import paddle
  from paddleslim.nas.ofa.layers import SuperEmbedding

  data = np.random.uniform(-1, 1, [32, 64]).astype('int64')
  config = {'channel': 16}
  emb = SuperEmbedding(64, 64)
  data = paddle.to_tensor(data)
  res = emb(data, **config)

.. py:class:: paddleslim.nas.ofa.layers.SuperBatchNorm2D(num_features, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format='NCHW', name=None):

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L937>`_

该接口用于构建 SuperBatchNorm2D 类的一个可调用对象。

**参数：**
  - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
  - **epsilon** (float, 可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
  - **momentum** (float, 可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。默认值：0.9。
  - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False, 则表示每个通道的伸缩固定为1，不可改变。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
  - **bias_attr** (ParamAttr, 可选) - 指定偏置参数属性的对象。如果为False, 则表示每一个通道的偏移固定为0，不可改变。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
  - **data_format** (string, 可选) - 指定输入数据格式，数据格式可以为"NCHW"。默认值：“NCHW”。
  - **name** (string, 可选) – BatchNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

**示例代码：**

.. code-block:: python

    import paddle
    import numpy as np
    from paddleslim.nas.ofa.layers import SuperBatchNorm2D

    np.random.seed(123)
    x_data = np.random.random(size=(2, 5, 2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    batch_norm = SuperBatchNorm2D(5)
    batch_norm_out = batch_norm(x)

.. py:class:: paddleslim.nas.ofa.layers.SuperInstanceNorm2D(num_features, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format='NCHW', name=None):

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L1004>`_

该接口用于构建 SuperInstanceNorm2D 类的一个可调用对象。

**参数：**
  - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
  - **epsilon** (float, 可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
  - **momentum** (float, 可选) - 本参数目前对 ``InstanceNorm2D`` 无效，无需设置。默认值：0.9。
  - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False, 则表示每个通道的伸缩固定为1，不可改变。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
  - **bias_attr** (ParamAttr, 可选) - 指定偏置参数属性的对象。如果为False, 则表示每一个通道的偏移固定为0，不可改变。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
  - **data_format** (string, 可选) - 指定输入数据格式，数据格式可以为"NCHW"。默认值：“NCHW”。
  - **name** (string, 可选) – BatchNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

**示例代码：**

.. code-block:: python

    import paddle
    import numpy as np
    from paddleslim.nas.ofa.layers import SuperInstanceNorm2D

    np.random.seed(123)
    x_data = np.random.random(size=(2, 5, 2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    instance_norm = SuperInstanceNorm2D(5)
    out = instance_norm(x)

.. py:class:: paddleslim.nas.ofa.layers.SuperLayerNorm(normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None):

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/nas/ofa/layers.py#L1062>`_

该接口用于构建 SuperLayerNorm 类的一个可调用对象。

**参数：**
  - **normalized_shape** (int 或 list 或 tuple) – 需规范化的shape，期望的输入shape为 ``[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`` 。如果是单个整数，则此模块将在最后一个维度上规范化（此时最后一维的维度需与该参数相同）。
  - **epsilon** (float, 可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
  - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False固定为1，不进行学习。默认值为None, 表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr, 可选) - 指定偏置参数属性的对象。如果为False固定为0，不进行学习。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **name** (string, 可选) – LayerNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

**示例代码：**

.. code-block:: python

    import paddle
    import numpy as np
    from paddleslim.nas.ofa.layers import SuperLayerNorm

    np.random.seed(123)
    x_data = np.random.random(size=(2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    layer_norm = SuperLayerNorm(x_data.shape[1])
    layer_norm_out = layer_norm(x)


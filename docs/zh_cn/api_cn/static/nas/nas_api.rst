NAS
========

搜索空间参数的配置
----------------------


通过参数配置搜索空间。更多搜索空间的使用可以参考: `search_space <https://paddlepaddle.github.io/PaddleSlim/api_cn/search_space.html>`_

**参数：**

- **input_size(int|None)**：- ``input_size`` 表示输入 ``feature map`` 的大小。 ``input_size`` 和 ``output_size`` 用来计算整个模型结构中下采样次数。

- **output_size(int|None)**：- ``output_size`` 表示输出feature map的大小。 ``input_size`` 和 ``output_size`` 用来计算整个模型结构中下采样次数。

- **block_num(int|None)**：- ``block_num`` 表示搜索空间中block的数量。

- **block_mask(list|None)**：- ``block_mask`` 是一组由0、1组成的列表，0表示当前block是normal block，1表示当前block是reduction block。reduction block表示经过这个block之后的feature map大小下降为之前的一半，normal block表示经过这个block之后feature map大小不变。如果设置了  ``block_mask`` ，则主要以 ``block_mask`` 为主要配置， ``input_size`` ， ``output_size`` 和 ``block_num`` 三种配置是无效的。

SANAS
------

.. py:class:: paddleslim.nas.SANAS(configs, server_addr=("", 8881), init_temperature=None, reduce_rate=0.85, init_tokens=None, search_steps=300, save_checkpoint='./nas_checkpoint', load_checkpoint=None, is_server=True)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/sa_nas.py#L36>`_

SANAS（Simulated Annealing Neural Architecture Search）是基于模拟退火算法进行模型结构搜索的算法，一般用于离散搜索任务。

**参数：**

- **configs(list<tuple>)** - 搜索空间配置列表，格式是 ``[(key, {input_size, output_size, block_num, block_mask})]`` 或者 ``[(key)]`` （MobileNetV2、MobilenetV1和ResNet的搜索空间使用和原本网络结构相同的搜索空间，所以仅需指定 ``key`` 即可）, ``input_size`` 和 ``output_size`` 表示输入和输出的特征图的大小， ``block_num`` 是指搜索网络中的block数量， ``block_mask`` 是一组由0和1组成的列表，0代表不进行下采样的block，1代表下采样的block。 更多paddleslim提供的搜索空间配置可以参考[Search Space](../search_space.md)。
- **server_addr(tuple)** - SANAS的地址，包括server的ip地址和端口号，如果ip地址为None或者为""的话则默认使用本机ip。默认：（"", 8881）。
- **init_temperature(float)** - 基于模拟退火进行搜索的初始温度。如果init_template为None而且init_tokens为None，则默认初始温度为10.0，如果init_template为None且init_tokens不为None，则默认初始温度为1.0。详细的温度设置可以参考下面的Note。默认：None。
- **reduce_rate(float)** - 基于模拟退火进行搜索的衰减率。详细的退火率设置可以参考下面的Note。默认：0.85。
- **init_tokens(list|None)** - 初始化token，若init_tokens为空，则SA算法随机生成初始化tokens。默认：None。
- **search_steps(int)** - 搜索过程迭代的次数。默认：300。
- **save_checkpoint(str|None)** - 保存checkpoint的文件目录，如果设置为None的话则不保存checkpoint。默认： ``./nas_checkpoint`` 。
- **load_checkpoint(str|None)** - 加载checkpoint的文件目录，如果设置为None的话则不加载checkpoint。默认：None。
- **is_server(bool)** - 当前实例是否要启动一个server。默认：True。

**返回：**
一个SANAS类的实例

**示例代码：**

.. code-block:: python

   import paddle
   from paddleslim.nas import SANAS
   config = [('MobileNetV2Space')]
   paddle.enable_static()
   sanas = SANAS(configs=config)

.. note::

  - 初始化温度和退火率的意义:

    - SA算法内部会保存一个基础token（初始化token可以自己传入也可以随机生成）和基础score（初始化score为-1），下一个token会在当前SA算法保存的token的基础上产生。在SA的搜索过程中，如果本轮的token训练得到的score大于SA算法中保存的score，则本轮的token一定会被SA算法接收保存为下一轮token产生的基础token。

    - 初始温度越高表示SA算法当前处的阶段越不稳定，本轮的token训练得到的score小于SA算法中保存的score的话，本轮的token和score被SA算法接收的可能性越大。

    - 初始温度越低表示SA算法当前处的阶段越稳定，本轮的token训练得到的score小于SA算法中保存的score的话，本轮的token和score被SA算法接收的可能性越小。

    - 退火率越大，表示SA算法收敛的越慢，即SA算法越慢到稳定阶段。

    - 退火率越低，表示SA算法收敛的越快，即SA算法越快到稳定阶段。

  - 初始化温度和退火率的设置: 

    - 如果原本就有一个较好的初始化token，想要基于这个较好的token来进行搜索的话，SA算法可以处于一个较为稳定的状态进行搜索r这种情况下初始温度可以设置的低一些，例如设置为1.0，退火率设置的大一些，例如设置为0.85。如果想要基于这个较好的token利用贪心算法进行搜索，即只有当本轮token训练得到的score大于SA算法中保存的score，SA算法才接收本轮token，则退火率可设置为一个极小的数字，例如设置为0.85 ** 10。

    - 初始化token如果是随机生成的话，代表初始化token是一个比较差的token，SA算法可以处于一种不稳定的阶段进行搜索，尽可能的随机探索所有可能得token，从而找到一个较好的token。初始温度可以设置的高一些，例如设置为1000，退火率相对设置的小一些。

.. 

   .. py:method:: next_archs()

   获取下一组模型结构。
   
   **返回：**
   返回模型结构实例的列表，形式为list。
   
   **示例代码：**

   .. code-block:: python

      import paddle
      from paddleslim.nas import SANAS
      config = [('MobileNetV2Space')]
      paddle.enable_static()
      sanas = SANAS(configs=config)
      input = paddle.static.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
      archs = sanas.next_archs()
      for arch in archs:
          output = arch(input)
          input = output
      print(output)
   
   .. py:method:: reward(score)

   把当前模型结构的得分情况回传。
   
   **参数：**
   
   - **score<float>:** - 当前模型的得分，分数越大越好。
   
   **返回：**
   模型结构更新成功或者失败，成功则返回 ``True`` ，失败则返回 ``False`` 。
   
   **示例代码：**

   .. code-block:: python

      import paddle
      from paddleslim.nas import SANAS
      config = [('MobileNetV2Space')]
      paddle.enable_static()
      sanas = SANAS(configs=config)
      archs = sanas.next_archs()
      
      ### 假设网络计算出来的score是1，实际代码中使用时需要返回真实score。
      score=float(1.0)
      sanas.reward(float(score))
   
   
   .. py:method:: tokens2arch(tokens)

   通过一组tokens得到实际的模型结构，一般用来把搜索到最优的token转换为模型结构用来做最后的训练。tokens的形式是一个列表，tokens映射到搜索空间转换成相应的网络结构，一组tokens对应唯一的一个网络结构。
   
   **参数：**
   
   - **tokens(list):** - 一组tokens。tokens的长度和范围取决于搜索空间。
   
   **返回：**
   根据传入的token得到一个模型结构实例列表。
   
   **示例代码：**

   .. code-block:: python

      import paddle
      from paddleslim.nas import SANAS
      config = [('MobileNetV2Space')]
      paddle.enable_static()
      sanas = SANAS(configs=config)
      input = paddle.static.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
      tokens = ([0] * 25)
      archs = sanas.tokens2arch(tokens)[0]
      print(archs(input))
   
   .. py:method:: current_info()

   返回当前token和搜索过程中最好的token和reward。
   
   **返回：**
   搜索过程中最好的token，reward和当前训练的token，形式为dict。
   
   **示例代码：**

   .. code-block:: python

      import paddle
      from paddleslim.nas import SANAS
      config = [('MobileNetV2Space')]
      paddle.enable_static()
      sanas = SANAS(configs=config)
      print(sanas.current_info())



RLNAS
------

.. py:class:: paddleslim.nas.RLNAS(key, configs, use_gpu=False, server_addr=("", 8881), is_server=True, is_sync=False, save_controller=None, load_controller=None, **kwargs)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/rl_nas.py>`_

RLNAS (Reinforcement Learning Neural Architecture Search）是基于强化学习算法进行模型结构搜索的算法。

**参数：**

- **key<str>** - 使用的强化学习Controller名称，目前paddleslim支持的有`LSTM`和`DDPG`，自定义强化学习Controller请参考 `自定义强化学习Controller <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/custom_rl_controller.md>`_
- **configs(list<tuple>)** - 搜索空间配置列表，格式是 ``[(key, {input_size, output_size, block_num, block_mask})]`` 或者 ``[(key)]`` （MobileNetV2、MobilenetV1和ResNet的搜索空间使用和原本网络结构相同的搜索空间，所以仅需指定 ``key`` 即可）, ``input_size`` 和 ``output_size`` 表示输入和输出的特征图的大小， ``block_num`` 是指搜索网络中的block数量， ``block_mask`` 是一组由0和1组成的列表，0代表不进行下采样的block，1代表下采样的block。 更多paddleslim提供的搜索空间配置可以参考[Search Space](../search_space.md)。
- **use_gpu(bool)** - 是否使用GPU来训练Controller。默认：False。
- **server_addr(tuple)** - RLNAS中Controller的地址，包括server的ip地址和端口号，如果ip地址为None或者为""的话则默认使用本机ip。默认：（"", 8881）。
- **is_server(bool)** - 当前实例是否要启动一个server。默认：True。
- **is_sync(bool)** - 是否使用同步模式更新Controller，该模式仅在多client下有差别。默认：False。
- **save_controller(str|None|False)** - 保存Controller的checkpoint的文件目录，如果设置为None的话则保存checkpoint到默认路径 ``./.rlnas_controller`` ，如果设置为False的话则不保存checkpoint。默认：None 。
- **load_controller(str|None)** - 加载Controller的checkpoint的文件目录，如果设置为None的话则不加载checkpoint。默认：None。
- **\*\*kwargs** - 附加的参数，由具体强化学习算法决定，`LSTM`和`DDPG`的附加参数请参考note。

.. note::

  - **`LSTM`算法的附加参数：**

    - lstm_num_layers(int, optional): - Controller中堆叠的LSTM的层数。默认：1.
    - hidden_size(int, optional): - LSTM中隐藏层的大小。默认：100.
    - temperature(float, optional): - 是否在计算每个token过程中做温度平均。默认：None.
    - tanh_constant(float, optional): 是否在计算每个token过程中做tanh激活，并乘上`tanh_constant`值。 默认：None。
    - decay(float, optional): LSTM中记录rewards的baseline的平滑率。默认：0.99.
    - weight_entropy(float, optional): 在更新controller参数时是否为接收到的rewards加上计算token过程中的带权重的交叉熵值。默认：None。
    - controller_batch_size(int, optional): controller的batch_size，即每运行一次controller可以拿到几组token。默认：1.
    - controller_lr(float, optional): controller的学习率，默认：1e-4。
    - controller_decay_steps(int, optional): controller学习率下降步长，设置为None的时候学习率不下降。默认：None。
    - controller_decay_rate(float, optional): controller学习率衰减率，默认：None。


  - **`DDPG`算法的附加参数：**

    **注意：** 使用`DDPG`算法的话必须安装parl。安装方法: `pip install parl`

    - obs_dim(int): observation的维度。
    - model(class，optional): DDPG算法中使用的具体的模型，一般是个类，包含actor_model和critic_model，需要实现两个方法，一个是policy用来获得策略，另一个是value，需要获得Q值。可以参考默认的 `default_model <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/common/RL_controller/DDPG/ddpg_model.py>`_  实现您自己的model。默认：`default_ddpg_model`.
    - actor_lr(float, optional): actor网络的学习率。默认：1e-4.
    - critic_lr(float, optional): critic网络的学习率。默认：1e-3.
    - gamma(float, optional): 接收到rewards之后的折扣因子。默认：0.99.
    - tau(float, optional): DDPG中把models的参数同步累积到target_model上时的折扣因子。默认：0.001.
    - memory_size(int, optional): DDPG中记录历史信息的池子大小。默认：10.
    - reward_scale(float, optional): 记录历史信息时，对rewards信息进行的折扣因子。默认：0.1.
    - controller_batch_size(int, optional): controller的batch_size，即每运行一次controller可以拿到几个token。默认：1.
    - actions_noise(class, optional): 通过DDPG拿到action之后添加的噪声，设置为False或者None时不添加噪声。默认：default_noise.
..

**返回：**
一个RLNAS类的实例

**示例代码：**

.. code-block:: python

   import paddle
   from paddleslim.nas import RLNAS
   config = [('MobileNetV2Space')]

   paddle.enable_static()
   rlnas = RLNAS(key='lstm', configs=config)


.. py:method:: next_archs(obs=None)

获取下一组模型结构。

**参数：**

- **obs<int|np.array>** - 需要获取的模型结构数量或者当前模型的observations。

**返回：**
返回模型结构实例的列表，形式为list。
 
**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim.nas import RLNAS
  config = [('MobileNetV2Space')]
  paddle.enable_static()
  rlnas = RLNAS(key='lstm', configs=config)
  input = paddle.static.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
  archs = rlnas.next_archs(1)[0]
  for arch in archs:
      output = arch(input)
      input = output
  print(output)

.. py:method:: reward(rewards, **kwargs):

把当前模型结构的rewards回传。

**参数：**

- **rewards<float|list<float>>:** - 当前模型的rewards，分数越大越好。
- **\*\*kwargs:** - 附加的参数，取决于具体的强化学习算法。

**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim.nas import RLNAS
  config = [('MobileNetV2Space')]
  paddle.enable_static()
  rlnas = RLNAS(key='lstm', configs=config)
  rlnas.next_archs(1)
  rlnas.reward(1.0)

.. note::
  reward这一步必须在`next_token`之后执行。
..

.. py:method:: final_archs(batch_obs):

获取最终的模型结构。一般在controller训练完成之后会获取几十个模型结构进行完整的实验。

**参数：**

- **obs<int|np.array>** - 需要获取的模型结构数量或者当前模型的observations。

**返回：**
返回模型结构实例的列表，形式为list。
 
**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim.nas import RLNAS
  config = [('MobileNetV2Space')]
  paddle.enable_static()
  rlnas = RLNAS(key='lstm', configs=config)
  archs = rlnas.final_archs(1)
  print(archs)

.. py:method:: tokens2arch(tokens):

通过一组tokens得到实际的模型结构，一般用来把搜索到最优的token转换为模型结构用来做最后的训练。tokens的形式是一个列表，tokens映射到搜索空间转换成相应的网络结构，一组tokens对应唯一的一个网络结构。

**参数：**

- **tokens(list):** - 一组tokens。tokens的长度和范围取决于搜索空间。

**返回：**
根据传入的token得到一个模型结构实例列表。

**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim.nas import RLNAS
  config = [('MobileNetV2Space')]
  paddle.enable_static()
  rlnas = RLNAS(key='lstm', configs=config)
  input = paddle.static.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
  tokens = ([0] * 25)
  archs = rlnas.tokens2arch(tokens)[0]
  print(archs(input))


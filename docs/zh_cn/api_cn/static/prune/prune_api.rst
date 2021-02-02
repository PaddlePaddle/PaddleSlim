卷积层通道剪裁
================

Pruner
----------

.. py:class:: paddleslim.prune.Pruner(criterion="l1_norm")

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/pruner.py#L28>`_

对卷积网络的通道进行一次剪裁。剪裁一个卷积层的通道，是指剪裁该卷积层输出的通道。卷积层的权重形状为 ``[output_channel, input_channel, kernel_size, kernel_size]`` ，通过剪裁该权重的第一纬度达到剪裁输出通道数的目的。

**参数：**

- **criterion** - 评估一个卷积层内通道重要性所参考的指标。目前支持 ``l1_norm`` , ``bn_scale`` , ``geometry_median``  。默认为 ``l1_norm`` 。若该参数设为 ``bn_scale`` , 则表示剪枝算法将根据卷积层后连接的BatchNorm层的Scale参数的绝对值大小作为评估卷积层内通道重要性所参考的指标。若参数设为 ``geometry_median``, 则表示剪枝算法将基于卷积层内通道的几何中心作为评估卷积层内通道重要性参考指标。 在初始化Pruner()类实例时，若没有传入该参数，则表示Pruner()使用criterion默认参数值 ``l1_norm`` ；可以显示地传入criterion的值以改变剪枝算法的剪枝策略。
- **idx_selector** - 基于卷积层内通道重要性分数，指示选择裁剪的卷积层内通道索引的策略。目前支持 ``default_idx_selector`` 和 ``optimal_threshold`` 两种选择策略。默认为 ``default_idx_selector`` 。 ``default_idx_selector`` 策略表示根据卷积层内通道的重要性分数进行选择要被裁剪的通道。 ``optimal_threshold`` 策略和 ``bn_scale`` 准则配合使用，即将 ``criterion`` 设置为 ``bn_scale`` ， 并将该参数设置为 ``optimal_threshold``,  表示根据卷积层后链接的BatchNorm层的Scale参数计算出要裁剪的最优裁剪阈值，并根据该阈值进行通道裁剪。在初始话Pruner()实例时，若没有传入该参数，则表示Pruner()使用idx_selector默认参数 ``default_idx_selector`` 。

**返回：** 一个Pruner类的实例

**示例代码：**

.. code-block:: python

   from paddleslim.prune import Pruner
   pruner = Pruner()       
..
 
   .. py:method:: paddleslim.prune.Pruner.prune(program, scope, params, ratios, place=None, lazy=False, only_graph=False, param_backup=False, param_shape_backup=False)

   对目标网络的一组卷积层的权重进行裁剪。
   
   **参数：**
   
   - **program(paddle.fluid.Program)** - 要裁剪的目标网络。更多关于Program的介绍请参考：`Program概念介绍 <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program>`_。
   
   - **scope(paddle.fluid.Scope)** - 要裁剪的权重所在的 ``scope`` ，Paddle中用 ``scope`` 实例存放模型参数和运行时变量的值。Scope中的参数值会被 ``inplace`` 的裁剪。更多介绍请参考: `Scope概念介绍 <>`_
   
   - **params(list<str>)** - 需要被裁剪的卷积层的参数的名称列表。可以通过以下方式查看模型中所有参数的名称:
   
   .. code-block:: python
   
      for block in program.blocks:
          for param in block.all_parameters():
              print("param: {}; shape: {}".format(param.name, param.shape))
   
   - **ratios(list<float>)** - 用于裁剪 ``params`` 的剪切率，类型为列表。该列表长度必须与 ``params`` 的长度一致。
   
   - **place(paddle.fluid.Place)** - 待裁剪参数所在的设备位置，可以是 ``CUDAPlace`` 或 ``CPUPlace`` 。[Place概念介绍]()
   
   - **lazy(bool)** - ``lazy`` 为True时，通过将指定通道的参数置零达到裁剪的目的，参数的 ``shape保持不变`` ； ``lazy`` 为False时，直接将要裁的通道的参数删除，参数的 ``shape`` 会发生变化。
   
   - **only_graph(bool)** - 是否只裁剪网络结构。在Paddle中，Program定义了网络结构，Scope存储参数的数值。一个Scope实例可以被多个Program使用，比如定义了训练网络的Program和定义了测试网络的Program是使用同一个Scope实例的。 ``only_graph`` 为True时，只对Program中定义的卷积的通道进行剪裁； ``only_graph`` 为false时，Scope中卷积参数的数值也会被剪裁。默认为False。
   
   - **param_backup(bool)** - 是否返回对参数值的备份。默认为False。
   
   - **param_shape_backup(bool)** - 是否返回对参数 ``shape`` 的备份。默认为False。
   
   **返回：**
   
   - **pruned_program(paddle.fluid.Program)** - 被裁剪后的Program。
   
   - **param_backup(dict)** - 对参数数值的备份，用于恢复Scope中的参数数值。
   
   - **param_shape_backup(dict)** - 对参数形状的备份。
   
   **示例：**
   
   点击 `AIStudio <https://aistudio.baidu.com/aistudio/projectDetail/200786>`_ 执行以下示例代码。

   .. code-block:: python
   
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
      # Initiallize Pruner() instance with default criterion and idx_selector
      pruner = Pruner()
      # Set criterion
      # criterion = 'geometry_median'
      # pruner = Pruner(criterion=criterion)
      # Set criterion and idx_selector
      # criterion = 'bn_scale'
      # idx_selector = 'optimal_threshold'
      # pruner = Pruner(criterion=criterion, idx_selector=idx_selector)
     
      main_program, _, _ = pruner.prune(
          main_program,
          scope,
          params=["conv4_weights"],
          ratios=[0.5],
          place=place,
          lazy=False,
          only_graph=False,
          param_backup=False,
          param_shape_backup=False)
      
      for param in main_program.global_block().all_parameters():
          if "weights" in param.name:
              print("param name: {}; param shape: {}".format(param.name, param.shape))
      

sensitivity
--------------

.. py:function:: paddleslim.prune.sensitivity(program, place, param_names, eval_func, sensitivities_file=None, pruned_ratios=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py>`_

计算网络中每个卷积层的敏感度。每个卷积层的敏感度信息统计方法为：依次剪掉当前卷积层不同比例的输出通道数，在测试集上计算剪裁后的精度损失。得到敏感度信息后，可以通过观察或其它方式确定每层卷积的剪裁率。

**参数：**

- **program(paddle.fluid.Program)** - 待评估的目标网络。更多关于Program的介绍请参考：`Program概念介绍 <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program>`_。

- **place(paddle.fluid.Place)** - 待分析的参数所在的设备位置，可以是 ``CUDAPlace`` 或 ``CPUPlace`` 。[Place概念介绍]()

- **param_names(list<str>)** - 待分析的卷积层的参数的名称列表。可以通过以下方式查看模型中所有参数的名称:

.. code-block:: python
   for block in program.blocks:
       for param in block.all_parameters():
           print("param: {}; shape: {}".format(param.name, param.shape))

- **eval_func(function)** - 用于评估裁剪后模型效果的回调函数。该回调函数接受被裁剪后的 ``program`` 为参数，返回一个表示当前program的精度，用以计算当前裁剪带来的精度损失。

- **sensitivities_file(str)** - 保存敏感度信息的本地文件系统的文件。在敏感度计算过程中，会持续将新计算出的敏感度信息追加到该文件中。重启任务后，文件中已有敏感度信息不会被重复计算。该文件可以用 ``pickle`` 加载。

- **pruned_ratios(list<float>)** - 计算卷积层敏感度信息时，依次剪掉的通道数比例。默认为 ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`` 。

**返回：**

- **sensitivities(dict)** - 存放敏感度信息的dict，其格式为：

.. code-block:: python

  {"weight_0":
     {0.1: 0.22,
      0.2: 0.33
     },
   "weight_1":
     {0.1: 0.21,
      0.2: 0.4
     }
  }

其中， ``weight_0`` 是卷积层参数的名称， ``sensitivities['weight_0']`` 的 ``value`` 为剪裁比例， ``value`` 为精度损失的比例。

**示例：**

点击 `AIStudio <https://aistudio.baidu.com/aistudio/projectdetail/201401>`_ 运行以下示例代码。

.. code-block:: python

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
   
   val_reader = paddle.fluid.io.batch(reader.test(), batch_size=128)
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
   

merge_sensitive
----------------

.. py:function:: paddleslim.prune.merge_sensitive(sensitivities)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py>`_

合并多个敏感度信息。

参数：

- **sensitivities(list<dict> | list<str>)** - 待合并的敏感度信息，可以是字典的列表，或者是存放敏感度信息的文件的路径列表。

返回：

- **sensitivities(dict)** - 合并后的敏感度信息。其格式为：

.. code-block:: bash

   {"weight_0":
      {0.1: 0.22,
       0.2: 0.33
      },
    "weight_1":
      {0.1: 0.21,
       0.2: 0.4
      }
   }
   

其中， ``weight_0`` 是卷积层参数的名称， ``sensitivities['weight_0']`` 的 ``value`` 为剪裁比例， ``value`` 为精度损失的比例。

示例：

.. code-block:: python

   from paddleslim.prune import merge_sensitive
   sen0 = {"weight_0":
      {0.1: 0.22,
       0.2: 0.33
      },
    "weight_1":
      {0.1: 0.21,
       0.2: 0.4
      }
   }
   sen1 = {"weight_0":
      {0.3: 0.41,
      },
    "weight_2":
      {0.1: 0.10,
       0.2: 0.35
      }
   }
   sensitivities = merge_sensitive([sen0, sen1])
   print(sensitivities)


load_sensitivities
---------------------

.. py:function:: paddleslim.prune.load_sensitivities(sensitivities_file)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py#L184>`_

从文件中加载敏感度信息。

参数：

- **sensitivities_file(str)** - 存放敏感度信息的本地文件.

返回：

- **sensitivities(dict)** - 敏感度信息。

示例：

.. code-block:: python

  import pickle
  from paddleslim.prune import load_sensitivities
  sen = {"weight_0":
     {0.1: 0.22,
      0.2: 0.33
     },
   "weight_1":
     {0.1: 0.21,
      0.2: 0.4
     }
  }
  sensitivities_file = "sensitive_api_demo.data"
  with open(sensitivities_file, 'wb') as f:
      pickle.dump(sen, f)
  sensitivities = load_sensitivities(sensitivities_file)
  print(sensitivities)

get_ratios_by_loss
-------------------

.. py:function:: paddleslim.prune.get_ratios_by_loss(sensitivities, loss)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py>`_

根据敏感度和精度损失阈值计算出一组剪切率。对于参数 ``w`` , 其剪裁率为使精度损失低于 ``loss`` 的最大剪裁率。

**参数：**

- **sensitivities(dict)** - 敏感度信息。

- **loss** - 精度损失阈值。

**返回：**

- **ratios(dict)** - 一组剪切率。 ``key`` 是待剪裁参数的名称。 ``value`` 是对应参数的剪裁率。

**示例：**

.. code-block:: python
   
  from paddleslim.prune import get_ratios_by_loss
  sen = {"weight_0":
     {0.1: 0.22,
      0.2: 0.33
     },
   "weight_1":
     {0.1: 0.21,
      0.2: 0.4
     }
  }
  
  ratios = get_ratios_by_loss(sen, 0.3)
  print(ratios)

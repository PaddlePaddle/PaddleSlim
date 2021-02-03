L2NormFilterPruner
==================

.. py:class:: paddleslim.L2NormFilterPruner(model, inputs, sen_file=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/paddleslim/dygraph/prune/l2norm_pruner.py>`_

用于剪裁卷积层输出通道的的剪裁器。该剪裁器按 ``Filters`` 的 ``l2-norm`` 统计值对单个卷积层内的 ``Filters`` 的重要性进行排序，并按指定比例剪裁掉相对不重要的 ``Filters`` 。对 ``Filters`` 的剪裁等价于剪裁卷积层的输出通道数。

**参数：**

- **model(paddle.nn.Layer)** - 待裁剪的动态图模型。

- **inputs(list<Object>)** - 待裁模型执行时需要的参数列表。该参数列表会以 ``model(*inputs)`` 的形式被调用。注意，被调用 ``model(*inputs)`` 使用到的卷积层才能被剪裁。以上调用用于将动态转为静态图，以便获得网络结构拓扑图，从而分析多个卷积层间的依赖关系。

- **sen_file(str)** - 存储敏感度信息的文件，需要指定为绝对路径。在调用当前剪裁器的 ``sensitive`` 方法时，敏感度信息会以增量的形式追加到文件 ``sen_file`` 中。如果用户不需要敏感度剪裁策略，可以将该选项设置为 ``None`` 。默认为None。

**返回：** 一个剪裁器实例。

**示例代码：**

.. code-block:: python

   from paddleslim import L2NormFilterPruner
   pruner = L2NormFilterPruner()       
..
 
   .. py:method:: prune_var(var_name, pruned_dims, pruned_ratio, apply="impretive")

   按指定的比例inplace地对原模型的单个卷积层及其相关卷积层进行剪裁。
   
   **参数：**
   
   - **var_name(str)** - 卷积层 ``weight`` 变量的名称。可以通过 `parameters API <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dygraph/layers/Layer_cn.html#parameters>`_ 获得所有卷积层 ``weight`` 变量的名称。
   
   - **pruned_dims(list<int>)** - 待废弃选项。卷积层 ``weight`` 变量中 ``Filter`` 数量所在轴，一般情况下设置为 ``[0]`` 即可。
   
   - **pruned_ratio(float)** - 待剪裁掉的输出通道比例。
   
   - **apply(str)** - 实施剪裁操作的类型，候选值为："impretive"，"lazy" 和 ``None`` 。如果设置为"impretive"，则会从输出特征图中真正的删掉待剪裁的通道。如果设置为"lazy"，则仅会将待剪裁的通道置为 ``0`` 。如果设置为None，则仅会生成并返回剪裁计划，并不会将剪裁真正实施到原模型上，剪裁计划相关信息存储在类型为 `paddleslim.PruningPlan <>`_ 的实例中。该选项默认为"impretive"。
   
   **返回：**
   
   - **plan(paddleslim.PruningPlan)** - 存储剪裁计划的实例。
   
   **示例：**
   
   点击 `AIStudio <>`_ 执行以下示例代码。

   .. code-block:: python

      from paddle.vision.models import mobilenet_v1
      from paddleslim import L2NormFilterPruner
      net = mobilenet_v1(pretrained=False) 
      pruner = L2NormFilterPruner(net, [1, 3, 224, 224])
      plan = pruner.prun_var("conv2d_26.w_0", [0])
      print(f"plan: {plan}")
      paddle.summary(net, (1, 3, 224, 224))
   
   ..  

   .. py:method:: prune_vars(ratios, axis, apply="impretive")

   按指定的比例inplace地对原模型的多个卷积层及其相关卷积层进行剪裁。
   
   **参数：**
   
   - **ratios(dict)** - 待剪裁卷积层 ``weight`` 变量名称以及对应的剪裁率。其中字典的 ``key`` 为 ``str`` 类型，为变量名称， ``value`` 为 ``float`` 类型，表示对应卷积层需要剪掉的输出通道的比例。可以通过 `parameters API <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dygraph/layers/Layer_cn.html#parameters>`_ 获得所有卷积层 ``weight`` 变量的名称。
   
   - **axis(list<int>)** - 待废弃选项。卷积层 ``weight`` 变量中 ``Filter`` 数量所在轴，一般情况下设置为 ``[0]`` 即可。
   
   - **apply(str)** - 实施剪裁操作的类型，候选值为："impretive"，"lazy" 和 ``None`` 。如果设置为"impretive"，则会从输出特征图中真正的删掉待剪裁的通道。如果设置为"lazy"，则仅会将待剪裁的通道置为 ``0`` 。如果设置为None，则仅会生成并返回剪裁计划，并不会将剪裁真正实施到原模型上，剪裁计划相关信息存储在类型为 `paddleslim.PruningPlan <>`_ 的实例中。该选项默认为"impretive"。
   
   **返回：**
   
   - **plan(paddleslim.PruningPlan)** - 存储剪裁计划的实例。
   
   **示例：**
   
   点击 `AIStudio <>`_ 执行以下示例代码。

   .. code-block:: python

      from paddle.vision.models import mobilenet_v1
      from paddleslim import L2NormFilterPruner
      net = mobilenet_v1(pretrained=False) 
      pruner = L2NormFilterPruner(net, [1, 3, 224, 224])
      plan = pruner.prun_vars({"conv2d_26.w_0": 0.5}, [0])
      print(f"plan: {plan}")
      paddle.summary(net, (1, 3, 224, 224))

   ..

   .. py:method:: sensitive(eval_func=None, sen_file=None, target_vars=None, skip_vars=[])

   计算或获得模型的敏感度信息。当所有选项为默认值时，该方法返回当前剪裁器已计算的敏感度信息。当选项被正确设置时，该方法会计算根据当前剪裁器的剪裁策略计算分析模型的敏感度信息，并将敏感度信息追加保存到指定的文件中，同时敏感度信息会缓存到当前剪裁器中，以供后续其它操作使用。
   
   **参数：**
   
   - **eval_func** - 用于评估当前剪裁器中模型精度的方法，其参数列表应该为空，并返回一个 ``float`` 类型的数值来表示模型的精度。如果设置为None，则不进行敏感度计算，返回当前剪裁器缓存的已计算好的敏感度信息。默认为None。
 
   - **sen_file(str)** - 存储敏感度信息的文件，需要指定为绝对路径。在调用当前剪裁器的 ``sensitive`` 方法时，敏感度信息会以增量的形式追加到文件 ``sen_file`` 中。如果设置为None，则不进行敏感度计算，返回当前剪裁器缓存的已计算好的敏感度信息。默认为None。默认为None。
   
   - **target_vars(list<str>)** - 变量名称列表，用于指定需要计算哪些卷积层的 ``weight`` 的敏感度。如果设置为None，则所有卷积层的敏感度都会被计算。默认为None。

   - **skip_vars(list<str>)** - 变量名称列表，用于指定哪些卷积层的 ``weight`` 不需要计算敏感度。如果设置为 ``[]`` ，则仅会默认跳过 ``depthwise_conv2d`` 和 ``conv2d_transpose``。默认为 ``[]`` 。
   
   **返回：**
   
   - **sensitivities(dict)** - 存储敏感信息的字典，示例如下：

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
      
      其中，``weight_0`` 是卷积层权重变量的名称， ``sensitivities['weight_0']`` 是一个字典， key是用 ``float`` 类型数值表示的剪裁率，value是对应剪裁率下整个模型的精度损失比例。
   
   **示例：**
   
   点击 `AIStudio <>`_ 执行以下示例代码。

   .. code-block:: python

      from paddle.vision.models import mobilenet_v1
      from paddleslim import L2NormFilterPruner
      import paddle.vision.transforms as T
      from paddle.static import InputSpec as Input
      # 构建模型
      net = mobilenet_v1(pretrained=False) 

      # 准备高层API
      inputs = [Input([None, 3, 224, 224], 'float32', name='image')]
      labels = [Input([None, 1], 'int64', name='label')]
      model = paddle.Model(net, inputs, labels)
      model.prepare(
        None,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

      # 准备评估数据
      transform = T.Compose([
                          T.Transpose(),
                          T.Normalize([127.5], [127.5])
                      ])
      train_dataset = paddle.vision.datasets.Cifar10(mode="train", backend="cv2",transform=transform)
      val_dataset = paddle.vision.datasets.Cifar10(mode="test", backend="cv2",transform=transform)

      # 准备评估方法
      def eval_fn():
          result = model.evaluate(
            val_dataset,
            batch_size=128)
          return result['acc_top1']

      # 敏感度分析
      pruner = L2NormFilterPruner(net, [1, 3, 224, 224])
      sen = pruner.sensitive(eval_func=eval_fn, sen_file="./sen.pickle")
      print(f"sen: {sen}")


   .. py:method:: sensitive_prune(pruned_flops, skip_vars=[], align=None)

   根据敏感度信息和模型整体的FLOPs剪裁比例，对模型中的卷积层进行inplace地剪裁，不同卷积层被裁掉的比例与其敏感度成反比。
   
   **参数：**
   
   - **pruned_flops(float)** - 模型整体的FLOPs被裁剪的目标比例。注意：最终FLOPs被裁剪掉的比例不一定完全等于 ``pruned_flops``。

   - **skip_vars(list<str>)** - 变量名称列表，用于指定哪些卷积层的 ``weight`` 不需要计算敏感度。如果设置为 ``[]`` ，则仅会默认跳过 ``depthwise_conv2d`` 和 ``conv2d_transpose``。默认为 ``[]`` 。

   - **align(None|int)** - 是否将剪裁后的通道数量对齐到指定数值的倍数。假设原通道数为32，剪裁比例为0.2，如果 ``align`` 为None，则剪裁后通道数为26；如果 ``align`` 为8，则剪裁后的通道数为24。默认为None。
 
   
   **返回：**
   
   - **plan(paddleslim.PruningPlan)** - 存储剪裁计划的实例。
   
   **示例：**
   
   点击 `AIStudio <>`_ 执行以下示例代码。

   .. code-block:: python

      from paddle.vision.models import mobilenet_v1
      from paddleslim import L2NormFilterPruner
      import paddle.vision.transforms as T
      from paddle.static import InputSpec as Input
      # 构建模型
      net = mobilenet_v1(pretrained=False) 

      # 准备高层API
      inputs = [Input([None, 3, 224, 224], 'float32', name='image')]
      labels = [Input([None, 1], 'int64', name='label')]
      model = paddle.Model(net, inputs, labels)
      model.prepare(
        None,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

      # 准备评估数据
      transform = T.Compose([
                          T.Transpose(),
                          T.Normalize([127.5], [127.5])
                      ])
      train_dataset = paddle.vision.datasets.Cifar10(mode="train", backend="cv2",transform=transform)
      val_dataset = paddle.vision.datasets.Cifar10(mode="test", backend="cv2",transform=transform)

      # 准备评估方法
      def eval_fn():
          result = model.evaluate(
            val_dataset,
            batch_size=128)
          return result['acc_top1']

      # 敏感度分析
      pruner = L2NormFilterPruner(net, [1, 3, 224, 224])
      sen = pruner.sensitive(eval_func=eval_fn, sen_file="./sen.pickle")
      plan = pruner.sensitive_prune(0.5, align=8)
      print(f"plan: {plan}")




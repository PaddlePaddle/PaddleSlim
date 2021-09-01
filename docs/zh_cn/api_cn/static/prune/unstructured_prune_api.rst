非结构化稀疏
================

UnstrucuturedPruner
----------

.. py:class:: paddleslim.prune.UnstructuredPruner(program, mode, ratio=0.55, threshold=1e-2, scope=None, place=None, skip_params_type, skip_params_func=None, configs=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/unstructured_pruner.py>`_

对于神经网络中的参数进行非结构化稀疏。非结构化稀疏是指，根据某些衡量指标，将不重要的参数置0。其不按照固定结构剪裁（例如一个通道等），这是和结构化剪枝的主要区别。

**参数：**

- **program(paddle.static.Program)** - 一个paddle.static.Program对象，是待剪裁的模型。
- **mode(str)** - 稀疏化的模式，目前支持的模式有：'ratio'和'threshold'。在'ratio'模式下，会给定一个固定比例，例如0.55，然后所有参数中重要性较低的50%会被置0。类似的，在'threshold'模式下，会给定一个固定阈值，例如1e-2，然后重要性低于1e-2的参数会被置0。
- **ratio(float)** - 稀疏化比例期望，只有在 mode=='ratio' 时才会生效。
- **threshold(float)** - 稀疏化阈值期望，只有在 mode=='threshold' 时才会生效。
- **scope(paddle.static.Scope)** - 一个paddle.static.Scope对象，存储了所有变量的数值，默认（None）时表示paddle.static.global_scope。
- **place(CPUPlace|CUDAPlace)** - 模型执行的设备，类型为CPUPlace或者CUDAPlace，默认（None）时代表CPUPlace。
- **skip_params_type(String)** - 用以指定哪些类型的参数跳过稀疏。目前只支持None和"exclude_conv1x1"两个选项，后者表示只稀疏化1x1卷积。
- **skip_params_func(function)** - 一个指向function的指针，该function定义了哪些参数不应该被剪裁，默认（None）时代表所有归一化层参数不参与剪裁。
- **configs(Dict)** - 传入额外的训练参数。在该class中，默认为None。在子类中，可以传入不同的参数控制不同的稀疏化训练策略，详情见GMPUnstructuredPruner。

**返回：** 一个UnstructuredPruner类的实例

**示例代码：**

.. code-block:: python

  import paddle
  import paddle.fluid as fluid
  from paddleslim.prune import UnstructuredPruner 

  paddle.enable_static()

  train_program = paddle.static.default_main_program()
  startup_program = paddle.static.default_startup_program()

  with fluid.program_guard(train_program, startup_program):
      image = fluid.data(name='x', shape=[None, 1, 28, 28])
      label = fluid.data(name='label', shape=[None, 1], dtype='int64')
      conv = fluid.layers.conv2d(image, 32, 1)
      feature = fluid.layers.fc(conv, 10, act='softmax')
      cost = fluid.layers.cross_entropy(input=feature, label=label)
      avg_cost = fluid.layers.mean(x=cost)

  place = paddle.static.cpu_places()[0]
  exe = paddle.static.Executor(place)
  exe.run(startup_program)

  pruner = UnstructuredPruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place)
..

  .. py:method:: paddleslim.prune.unstructured_pruner.UnstructuredPruner.step()

  更新稀疏化的阈值，如果是'threshold'模式，则维持设定的阈值，如果是'ratio'模式，则根据优化后的模型参数和设定的比例，重新计算阈值。

  **示例代码：**

  .. code-block:: python

    import paddle
    import paddle.fluid as fluid 
    from paddleslim.prune import UnstructuredPruner

    paddle.enable_static()

    train_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='x', shape=[None, 1, 28, 28])
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        conv = fluid.layers.conv2d(image, 32, 1)
        feature = fluid.layers.fc(conv, 10, act='softmax')
        cost = fluid.layers.cross_entropy(input=feature, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    place = paddle.static.cpu_places()[0]
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    pruner = UnstructuredPruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place)
    print(pruner.threshold)
    pruner.step()
    print(pruner.threshold) # 可以看出，这里的threshold和上面打印的不同，这是因为step函数根据设定的ratio更新了threshold数值，便于剪裁操作。
  ..

  .. py:method:: paddleslim.prune.unstructured_pruner.UnstructuredPruner.update_params()

  每一步优化后，重制模型中本来是0的权重。这一步通常用于模型evaluation和save之前，确保模型的稀疏率。但是，在训练过程中，由于前向过程中插入了稀疏化权重的op，故不需要开发者在训练过程中额外调用了。

  **示例代码：**

  .. code-block:: python

    import paddle
    import paddle.fluid as fluid
    from paddleslim.prune import UnstructuredPruner

    paddle.enable_static()

    train_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='x', shape=[None, 1, 28, 28])
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        conv = fluid.layers.conv2d(image, 32, 1)
        feature = fluid.layers.fc(conv, 10, act='softmax')
        cost = fluid.layers.cross_entropy(input=feature, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    place = paddle.static.cpu_places()[0]
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    pruner = UnstructuredPruner(paddle.static.default_main_program(), 'threshold', threshold=0.55, place=place)
    sparsity = UnstructuredPruner.total_sparse(paddle.static.default_main_program())
    print(sparsity)
    pruner.step()
    pruner.update_params()
    sparsity = UnstructuredPruner.total_sparse(paddle.static.default_main_program())
    print(sparsity) # 可以看出，这里打印的模型稀疏度与上述不同，这是因为update_params()函数置零了所有绝对值小于0.55的权重。

  ..

  .. py:method:: paddleslim.prune.unstructured_pruner.UnstructuredPruner.total_sparse(program)

  UnstructuredPruner中的静态方法，用于计算给定的模型（program）的稀疏度并返回。该方法为静态方法，是考虑到在单单做模型评价的时候，我们就不需要初始化一个UnstructuredPruner示例了。

  **参数：**

  -  **program(paddle.static.Program)** - 要计算稠密度的目标网络。

  **返回：**
  
  - **sparsity(float)** - 模型的稀疏度。

  **示例代码：**

  .. code-block:: python

    import paddle
    import paddle.fluid as fluid
    from paddleslim.prune import UnstructuredPruner

    paddle.enable_static()

    train_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='x', shape=[None, 1, 28, 28])
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        conv = fluid.layers.conv2d(image, 32, 1)
        feature = fluid.layers.fc(conv, 10, act='softmax')
        cost = fluid.layers.cross_entropy(input=feature, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    place = paddle.static.cpu_places()[0]
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    sparsity = UnstructuredPruner.total_sparse(paddle.static.default_main_program())
    print(sparsity)

  ..

  .. py:method:: paddleslim.prune.unstructured_pruner.UnstructuredPruner.total_sparse_conv1x1(program)

  UnstructuredPruner中的静态方法，用于计算给定的模型（program）的1x1卷积稀疏度并返回。该方法为静态方法，是考虑到在单单做模型评价的时候，我们就不需要初始化一个UnstructuredPruner示例了。

  **参数：**

  -  **program(paddle.static.Program)** - 要计算稠密度的目标网络。

  **返回：**
  
  - **sparsity(float)** - 模型的1x1卷积部分的稀疏度。

  **示例代码：**

  .. code-block:: python

    import paddle
    import paddle.fluid as fluid
    from paddleslim.prune import UnstructuredPruner

    paddle.enable_static()

    train_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='x', shape=[None, 1, 28, 28])
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        conv1x1 = fluid.layers.conv2d(image, 32, 1)
        conv3x3 = fluid.layers.conv2d(conv1x1, 32, 3)
        feature = fluid.layers.fc(conv3x3, 10, act='softmax')
        cost = fluid.layers.cross_entropy(input=feature, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    place = paddle.static.cpu_places()[0]
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    sparsity = UnstructuredPruner.total_sparse_conv1x1(paddle.static.default_main_program())
    print(sparsity)

  ..

  .. py:method:: paddleslim.prune.unstructured_pruner.UnstructuredPruner.summarize_weights(program, ratio=0.1)

  该函数用于估计预训练模型中参数的分布情况，尤其是在不清楚如何设置threshold的数值时，尤为有用。例如，当输入为ratio=0.1时，函数会返回一个数值v，而绝对值小于v的权重的个数占所有权重个数的(100*ratio%)。

  **参数：**

  - **program(paddle.static.Program)** - 要分析权重分布的目标网络。
  - **ratio(float)** - 需要查看的比例情况，具体如上方法描述。

  **返回：**

  - **threshold(float)** - 和输入ratio对应的阈值。开发者可以根据该阈值初始化UnstructuredPruner。

  **示例代码：**

  .. code-block:: python

    import paddle
    import paddle.fluid as fluid
    from paddleslim.prune import UnstructuredPruner

    paddle.enable_static()

    train_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='x', shape=[None, 1, 28, 28])
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        conv = fluid.layers.conv2d(image, 32, 1)
        feature = fluid.layers.fc(conv, 10, act='softmax')
        cost = fluid.layers.cross_entropy(input=feature, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    place = paddle.static.cpu_places()[0]
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    pruner = UnstructuredPruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place)
    threshold = pruner.summarize_weights(paddle.static.default_main_program(), ratio=0.55)
    print(threshold)

  ..

GMPUnstrucuturedPruner
----------

.. py:class:: paddleslim.prune.GMPUnstructuredPruner(program, mode, ratio=0.55, threshold=1e-2, scope=None, place=None, skip_params_type=None, skip_params_func=None, configs=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/unstructured_pruner.py>`_

该类是UnstructuredPruner的一个子类，通过覆盖step()方法，优化了训练策略，使稀疏化训练更易恢复到稠密模型精度。其他方法均继承自父类。

**参数：**

- **program(paddle.static.Program)** - 一个paddle.static.Program对象，是待剪裁的模型。
- **mode(str)** - 稀疏化的模式，在这个类中，只有'ratio'模式被支持。
- **ratio(float)** - 稀疏化比例期望，只有在 mode=='ratio' 时才会生效。
- **threshold(float)** - 稀疏化阈值期望，只有在 mode=='threshold' 时才会生效。
- **scope(paddle.static.Scope)** - 一个paddle.static.Scope对象，存储了所有变量的数值，默认（None）时表示paddle.static.global_scope。
- **place(CPUPlace|CUDAPlace)** - 模型执行的设备，类型为CPUPlace或者CUDAPlace，默认（None）时代表CPUPlace。
- **skip_params_type(String)** - 用以指定哪些类型的参数跳过稀疏。目前只支持None和"exclude_conv1x1"两个选项，后者表示只稀疏化1x1卷积。
- **skip_params_func(function)** - 一个指向function的指针，该function定义了哪些参数不应该被剪裁，默认（None）时代表所有归一化层参数不参与剪裁。
- **configs(Dict)** - 传入额外的训练超参用以指导GMP训练过程。 

**返回：** 一个GMPUnstructuredPruner类的实例

**示例代码：**

.. code-block:: python

  import paddle
  import paddle.fluid as fluid
  from paddleslim.prune import GMPUnstructuredPruner

  paddle.enable_static()

  train_program = paddle.static.default_main_program()
  startup_program = paddle.static.default_startup_program()

  with fluid.program_guard(train_program, startup_program):
      image = fluid.data(name='x', shape=[None, 1, 28, 28])
      label = fluid.data(name='label', shape=[None, 1], dtype='int64')
      conv = fluid.layers.conv2d(image, 32, 1)
      feature = fluid.layers.fc(conv, 10, act='softmax')
      cost = fluid.layers.cross_entropy(input=feature, label=label)
      avg_cost = fluid.layers.mean(x=cost)

  place = paddle.static.cpu_places()[0]
  exe = paddle.static.Executor(place)
  exe.run(startup_program)

  configs = {
    'pruning_strategy': 'gmp', # pruning_strategy必须是'gmp'。
    'stable_iterations': 0,
    'pruning_iterations': 1000,
    'tunning_iterations': 1000,
    'resume_iteration': 0,
    'pruning_steps': 10,
    'initial_ratio': 0.15,
  }
  pruner = GMPUnstructuredPruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place, configs=configs)

  for i in range(2000):
    pruner.step()
    print(pruner.ratio) # 可以看到ratio从0.15非线性的增加到0.55。
..

  .. py:method:: paddleslim.prune.unstructured_pruner.GMPUnstructuredPruner.step()

  根据优化后的模型参数和设定的比例，重新计算阈值，并且更新mask。

  **示例代码：**

  .. code-block:: python

    import paddle
    import paddle.fluid as fluid 
    from paddleslim.prune import GMPUnstructuredPruner

    paddle.enable_static()

    train_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with fluid.program_guard(train_program, startup_program):
        image = fluid.data(name='x', shape=[None, 1, 28, 28])
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        conv = fluid.layers.conv2d(image, 32, 1)
        feature = fluid.layers.fc(conv, 10, act='softmax')
        cost = fluid.layers.cross_entropy(input=feature, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    place = paddle.static.cpu_places()[0]
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    configs = {
      'pruning_strategy': 'gmp', # pruning_strategy必须是'gmp'。
      'stable_iterations': 0,
      'pruning_iterations': 1000,
      'tunning_iterations': 1000,
      'resume_iteration': 0,
      'pruning_steps': 10,
      'initial_ratio': 0.15,
    }

    pruner = GMPUnstructuredPruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place, configs=configs)
    print(pruner.threshold)
    for i in range(200):
        pruner.step()
    print(pruner.threshold) # 可以看出，这里的threshold和上面打印的不同，这是因为step函数根据设定的ratio更新了threshold数值，便于剪裁操作。
  ..

make_unstructured_pruner
----------

.. py:method:: paddleslim.prune.make_unstructured_pruner(program, mode, ratio=0.55, threshold=1e-2, scope=None, place=None, skip_params_type=None, skip_params_func=None, configs=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/unstructured_pruner.py>`_

这是构造非结构化稀疏示例的入口函数，由于可以选择是否GMP训练策略，所以我们引入了这个函数，输入和父类UnstructuredPruner一致。

**参数：**

- **program(paddle.static.Program)** - 一个paddle.static.Program对象，是待剪裁的模型。
- **mode(str)** - 稀疏化的模式，目前支持的模式有：'ratio'和'threshold'。在'ratio'模式下，会给定一个固定比例，例如0.55，然后所有参数中重要性较低的50%会被置0。类似的，在'threshold'模式下，会给定一个固定阈值，例如1e-2，然后重要性低于1e-2的参数会被置0。
- **ratio(float)** - 稀疏化比例期望，只有在 mode=='ratio' 时才会生效。
- **threshold(float)** - 稀疏化阈值期望，只有在 mode=='threshold' 时才会生效。
- **scope(paddle.static.Scope)** - 一个paddle.static.Scope对象，存储了所有变量的数值，默认（None）时表示paddle.static.global_scope。
- **place(CPUPlace|CUDAPlace)** - 模型执行的设备，类型为CPUPlace或者CUDAPlace，默认（None）时代表CPUPlace。
- **skip_params_type(String)** - 用以指定哪些类型的参数跳过稀疏。目前只支持None和"exclude_conv1x1"两个选项，后者表示只稀疏化1x1卷积。
- **skip_params_func(function)** - 一个指向function的指针，该function定义了哪些参数不应该被剪裁，默认（None）时代表所有归一化层参数不参与剪裁。
- **configs(Dict)** - 传入额外的训练参数。在该class中，默认为None。在子类中，可以传入不同的参数控制不同的稀疏化训练策略，详情见GMPUnstructuredPruner。

**返回：** 一个UnstructuredPruner类的实例

**示例代码：**

.. code-block:: python

  import paddle
  import paddle.fluid as fluid
  from paddleslim.prune import make_unstructured_pruner

  paddle.enable_static()

  train_program = paddle.static.default_main_program()
  startup_program = paddle.static.default_startup_program()

  with fluid.program_guard(train_program, startup_program):
      image = fluid.data(name='x', shape=[None, 1, 28, 28])
      label = fluid.data(name='label', shape=[None, 1], dtype='int64')
      conv = fluid.layers.conv2d(image, 32, 1)
      feature = fluid.layers.fc(conv, 10, act='softmax')
      cost = fluid.layers.cross_entropy(input=feature, label=label)
      avg_cost = fluid.layers.mean(x=cost)

  place = paddle.static.cpu_places()[0]
  exe = paddle.static.Executor(place)
  exe.run(startup_program)

  # 构建pruner
  pruner = make_unstructured_pruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place)

  # 构建GMP pruner
  configs = {
      'pruning_strategy': 'gmp', # pruning_strategy必须是'gmp'。
      'stable_iterations': 0,
      'pruning_iterations': 1000,
      'tunning_iterations': 1000,
      'resume_iteration': 0,
      'pruning_steps': 10,
      'initial_ratio': 0.15,
  }
  pruner = make_unstructured_pruner(paddle.static.default_main_program(), 'ratio', ratio=0.55, place=place, configs=configs)

..


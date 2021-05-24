非结构化稀疏
================

UnstructuredPruner
----------

.. py:class:: paddleslim.UnstructuredPruner(model, mode, threshold=0.01, ratio=0.3, skip_params_func=None)


`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dygraph/prune/unstructured_pruner.py>`_

对于神经网络中的参数进行非结构化稀疏。非结构化稀疏是指，根据某些衡量指标，将不重要的参数置0。其不按照固定结构剪裁（例如一个通道等），这是和结构化剪枝的主要区别。

**参数：**

- **model(paddle.nn.Layer)** - 待剪裁的动态图模型。
- **mode(str)** - 稀疏化的模式，目前支持的模式有：'ratio'和'threshold'。在'ratio'模式下，会给定一个固定比例，例如0.5，然后所有参数中重要性较低的50%会被置0。类似的，在'threshold'模式下，会给定一个固定阈值，例如1e-5，然后重要性低于1e-5的参数会被置0。
- **ratio(float)** - 稀疏化比例期望，只有在 mode=='ratio' 时才会生效。
- **threshold(float)** - 稀疏化阈值期望，只有在 mode=='threshold' 时才会生效。
- **skip_params_func(function)** - 一个指向function的指针，该function定义了哪些参数不应该被剪裁，默认（None）时代表所有归一化层参数不参与剪裁。

**返回：** 一个UnstructuredPruner类的实例。

**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim import UnstructuredPruner
  from paddle.vision.models import LeNet as net
  import numpy as np

  place = paddle.set_device('cpu')
  model = net(num_classes=10)

  pruner = UnstructuredPruner(model, mode='ratio', ratio=0.5)

..

  .. py:method:: paddleslim.UnstructuredPruner.step()

  更新稀疏化的阈值，如果是'threshold'模式，则维持设定的阈值，如果是'ratio'模式，则根据优化后的模型参数和设定的比例，重新计算阈值。

  **示例代码：**

  .. code-block:: python

    import paddle
    from paddleslim import UnstructuredPruner
    from paddle.vision.models import LeNet as net
    import numpy as np

    place = paddle.set_device('cpu')
    model = net(num_classes=10)
    pruner = UnstructuredPruner(model, mode='ratio', ratio=0.5)

    print(pruner.threshold)
    pruner.step()
    print(pruner.threshold) # 可以看出，这里的threshold和上面打印的不同，这是因为step函数根据设定的ratio更新了threshold数值，便于剪裁操作。

  ..

  .. py:method:: paddleslim.UnstructuredPruner.update_params()

  每一步优化后，重制模型中本来是0的权重。这一步通常用于模型evaluation和save之前，确保模型的稀疏率。

  **示例代码：**

  .. code-block:: python

    import paddle
    from paddleslim import UnstructuredPruner
    from paddle.vision.models import LeNet as net
    import numpy as np

    place = paddle.set_device('cpu')
    model = net(num_classes=10)
    pruner = UnstructuredPruner(model, mode='threshold', threshold=0.5)

    density = UnstructuredPruner.total_sparse(model)
    print(density)
    model(paddle.to_tensor(
                np.random.uniform(0, 1, [16, 1, 28, 28]), dtype='float32'))
    pruner.update_params()
    density = UnstructuredPruner.total_sparse(model)
    print(density) # 可以看出，这里打印的模型稠密度与上述不同，这是因为update_params()函数置零了所有绝对值小于0.5的权重。

  ..

  ..  py:method:: paddleslim.UnstructuredPruner.total_sparse(model)

  UnstructuredPruner中的静态方法，用于计算给定的模型（model）的稠密度（1-稀疏度）并返回。该方法为静态方法，是考虑到在单单做模型评价的时候，我们就不需要初始化一个UnstructuredPruner示例了。

  **参数：**

  -  **model(paddle.nn.Layer)** - 要计算稠密度的目标网络。

  **返回：**
  
  - **density(float)** - 模型的稠密度。

  **示例代码：**

  .. code-block:: python

    import paddle
    from paddleslim import UnstructuredPruner
    from paddle.vision.models import LeNet as net
    import numpy as np

    place = paddle.set_device('cpu')
    model = net(num_classes=10)
    density = UnstructuredPruner.total_sparse(model)
    print(density)
    
  ..

  .. py:method:: paddleslim.UnstructuredPruner.summarize_weights(model, ratio=0.1)

  该函数用于估计预训练模型中参数的分布情况，尤其是在不清楚如何设置threshold的数值时，尤为有用。例如，当输入为ratio=0.1时，函数会返回一个数值v，而绝对值小于v的权重的个数占所有权重个数的(100*ratio%)。

  **参数：**

  - **model(paddle.nn.Layer)** - 要分析权重分布的目标网络。
  - **ratio(float)** - 需要查看的比例情况，具体如上方法描述。

  **返回：**

  - **threshold(float)** - 和输入ratio对应的阈值。开发者可以根据该阈值初始化UnstructuredPruner。

  **示例代码：**

  .. code-block:: python

    import paddle
    from paddleslim import UnstructuredPruner
    from paddle.vision.models import LeNet as net
    import numpy as np

    place = paddle.set_device('cpu')
    model = net(num_classes=10)
    pruner = UnstructuredPruner(model, mode='ratio', ratio=0.5)

    threshold = pruner.summarize_weights(model, 0.5)
    print(threshold)

  ..

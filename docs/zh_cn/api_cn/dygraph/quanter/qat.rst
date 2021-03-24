QAT
==================

.. py:class:: paddleslim.QAT(config=None, weight_preprocess=None, act_preprocess=None, weight_quantize=None, act_quantize=None)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dygraph/quant/quanter.py>`_

使用量化训练方法（Quant Aware Training, QAT）得到模拟量化模型，在需要量化的算子前插入模拟量化节点，为其激活和权重输入提前执行`量化-反量化`逻辑。


**参数：**

- **config(dict, Optional)** - 量化配置表。 默认为None，表示使用默认配置, 默认配置参考下方文档。

- **weight_preprocess(class, Optional)** - 自定义在对权重做量化之前，对权重进行处理的逻辑。这一接口只用于实验不同量化方法，验证量化训练效果。默认为None, 表示不对权重做任何预处理。

- **act_preprocess(class, Optional)** - 自定义在对激活值做量化之前，对激活值进行处理的逻辑。这一接口只用于实验不同量化方法，验证量化训练效果。默认为None, 表示不对激活值做任何预处理。

- **weight_quantize(class, Optional)** - 自定义对权重量化的方法。这一接口只用于实验不同量化方法，验证量化训练效果。默认为None, 表示使用默认量化方法。

- **act_quantize(class, Optional)** - 自定义对激活值量化的方法。这一接口只用于实验不同量化方法，验证量化训练效果。默认为None, 表示使用默认量化方法。


**返回：** 量化训练器实例。

**示例代码：**

.. code-block:: python

   from paddleslim import QAT
   quanter = QAT()
..


量化训练方法的参数配置

.. code-block:: python

    {
        # weight预处理方法，默认为None，代表不进行预处理；当需要使用`PACT`方法时设置为`"PACT"`
        'weight_preprocess_type': None,

        # activation预处理方法，默认为None，代表不进行预处理`
        'activation_preprocess_type': None,

        # weight量化方法, 默认为'channel_wise_abs_max', 此外还支持'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',

        # activation量化方法, 默认为'moving_average_abs_max', 此外还支持'abs_max'
        'activation_quantize_type': 'moving_average_abs_max',

        # weight量化比特数, 默认为 8
        'weight_bits': 8,

        # activation量化比特数, 默认为 8
        'activation_bits': 8,

        # 'moving_average_abs_max'的滑动平均超参, 默认为0.9
        'moving_rate': 0.9,

        # 需要量化的算子类型
        'quantizable_layer_type': ['Conv2D', 'Linear'],
    }
..

 
   .. py:method:: quantize(model)

   inplace地对模型进行量化训练前的处理，插入量化-反量化节点。
   
   **参数：**
   
   - **model(paddle.nn.Layer)** - 一个paddle Layer的实例，需要包含支持量化的算子，如：`Conv, Linear`
   
   
   **示例：**
   

   .. code-block:: python

      from paddle.vision.models import mobilenet_v1
      from paddleslim import QAT
      net = mobilenet_v1(pretrained=False) 
      quant_config = {
          'activation_preprocess_type': 'PACT',
          'quantizable_layer_type': ['Conv2D', 'Linear'],
      }
      quanter = QAT(config=quant_config)
      quanter.quantize(lenet)
      paddle.summary(net, (1, 3, 224, 224))
   
   ..  

   .. py:method:: save_quantized_model(model, path, input_spec=None)

   将指定的动态图量化模型导出为静态图预测模型，用于预测部署。
   
   量化预测模型可以使用`netron`软件打开，进行可视化查看。该量化预测模型和普通FP32预测模型一样，可以使用PaddleLite和PaddleInference加载预测，具体请参考`推理部署`章节。
   
   **参数：**
   
   - **model(paddle.nn.Layer)** - 量化训练结束，需要导出的量化模型，该模型由`quantize`接口产出。
   
   - **path(str)** - 导出的量化预测模型保存的路径，导出后在该路径下可以找到`model`和`params`文件。
   
   - **input_spec(list[InputSpec|Tensor], Optional)** - 描述存储模型forward方法的输入，可以通过InputSpec或者示例Tensor进行描述。如果为 None ，所有原 Layer forward方法的输入变量将都会被配置为存储模型的输入变量。默认为 None。
   
   
   **示例：**
   

   .. code-block:: python

      from paddle.vision.models import mobilenet_v1
      from paddleslim import QAT
      net = mobilenet_v1(pretrained=False) 
      quant_config = {
          'activation_preprocess_type': 'PACT',
          'quantizable_layer_type': ['Conv2D', 'Linear'],
      }
      quanter = QAT(config=quant_config)
      quanter.quantize(lenet)
      paddle.summary(net, (1, 3, 224, 224))

      quanter.save_quantized_model(
          net,
          './quant_model',
          input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])

   ..


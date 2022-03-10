延时预估器
================

TableLatencyPredictor
---------------------

.. py:class:: paddleslim.TableLatencyPredictor(table_file)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/analysis/latency_predictor.py>`_

延时预估器用于预估模型在特定硬件设备上的推理延时。在无需部署模型到实际环境的情况下，可以快速预估出多种部署环境和设置下的推理延时。

**参数：**

- **table_file(str)** -  指定硬件设备，可选“SD625”、“SD710”、“SD845”；或是传入已有的延时表路径。

**返回：** 一个TableLatencyPredictor类的实例。

**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim import TableLatencyPredictor
  
  predictor = TableLatencyPredictor(table_file='SD710')

..

  .. py:method:: paddleslim.TableLatencyPredictor.predict(model_file, param_file, data_type, threads, input_shape)

  预估模型在指定硬件设备上的延时。

  **参数：**

  -  **model_file(str)** - 推理模型的模型文件路径。
  -  **param_file(str)** - 推理模型的参数文件路径。
  -  **data_type(str)** - 推理模型的数据类型：‘fp32’或‘int8’。
  -  **threads(int)** - 设置预估多少线程数下的延时。目前只支持4线程，后续将支持更多线程数。
  -  **input_shape(list)** - 当模型为可变长输入时，该参数设置其输入形状。目前，暂不支持使用该参数控制模型输入，需在保存推理模型时设置确切的输入形状。

  **返回：** 

  -  **latency(float)** - 推理模型在指定设备上的延时。

  **示例代码：**

  .. code-block:: python

    import paddle
    from paddleslim import TableLatencyPredictor
    from paddle.vision.models import mobilenet_v1 
    from paddle.static import InputSpec

    predictor = TableLatencyPredictor(table_file='SD710')

    model = mobilenet_v1() 
    x_spec = InputSpec(shape=[1, 3, 224, 224], dtype='float32', name='inputs') 
    static_model = paddle.jit.to_static(model, input_spec=[x_spec]) 
    paddle.jit.save(static_model, 'mobilenet_v1') 
    
    latency = predictor.predict(model_file='mobilenet_v1.pdmodel', 
                                param_file='mobilenet_v1.pdiparams',
                                data_type='fp32')
    print("predicted latency:", latency)

  ..
早停算法
========
早停算法接口在实验中如何使用

MedianStop
------

.. py:class:: paddleslim.nas.early_stop.MedianStop(strategy, start_epoch, mode)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/early_stop/median_stop/median_stop.py>`_

MedianStop是利用历史较好实验的中间结果来判断当前实验是否有运行完成的必要，如果当前实验在中间步骤的结果差于历史记录的实验列表中相同步骤的结果的中值，则代表当前实验是较差的实验，可以提前终止。参考 `Google Vizier: A Service for Black-Box Optimization <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf>`_.

**参数：**

- **strategy<class instance>** - 搜索策略的实例，例如是SANAS的实例。
- **start_epoch<int>** - 起始epoch，代表从第几个epoch开始监控实验中间结果。
- **mode<str>** - 中间结果是越大越好还是越小越好，在'minimize'和'maxmize'之间选择。默认：'maxmize'。

**返回：**
一个MedianStop的实例

**示例代码：**

.. code-block:: python

  from paddleslim.nas import SANAS
  from paddleslim.nas.early_stop import MedianStop
  config = [('MobileNetV2Space')]
  sanas = SANAS(config, server_addr=("", 8732), save_checkpoint=None)
  earlystop = MedianStop(sanas, start_epoch = 2)

.. py:method:: get_status(step, result, epochs):

获取当前实验当前result的状态。

**参数：**

- **step<int>** - 当前实验是当前client中的第几个实验。
- **result<float>** - 当前实验的中间步骤的result，可以为损失值，也可以为准确率等指标，只要和`mode`对应即可。
- **epochs<int>** - 在搜索过程中每个实验需要运行的总得epoch数量。

**返回：**
返回当前实验在当前epoch的状态，为`GOOD`或者`BAD`，如果为`BAD`，则代表当前实验可以早停。

**示例代码：**

.. code-block:: python

  import paddle
  from paddleslim.nas import SANAS
  from paddleslim.nas.early_stop import MedianStop
  steps = 10
  epochs = 7
  
  config = [('MobileNetV2Space')]
  sanas = SANAS(config, server_addr=("", 8732), save_checkpoint=None)
  earlystop = MedianStop(sanas, 2)
  ### 假设网络中计算出来的loss是1.0，实际使用时需要获取真实的loss或者rewards。
  avg_loss = 1.0
  
  ### 假设我们要获取的是当前实验第7个epoch的状态，实际使用时需要传入真实要获取的steps和实验真实所处的epochs。
  status = earlystop.get_status(steps, avg_loss, epochs)
  print(status)

可微分模型架构搜索DARTS
=========

DARTSearch
---------

.. py:class:: paddleslim.nas.DARTSearch(model, train_reader, valid_reader, place, learning_rate=0.025, batchsize=64, num_imgs=50000, arch_learning_rate=3e-4, unrolled=False, num_epochs=50, epochs_no_archopt=0, use_data_parallel=False, save_dir='./', log_freq=50)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/1.1.0/paddleslim/nas/darts/train_search.py>`_

定义一个DARTS搜索示例，用于在特定数据集和搜索空间上启动模型架构搜索。

**参数：**

- **model** (Paddle Dygraph model)-用于搜索的超网络模型，需要以PaddlePaddle动态图的形式定义。
- **train_reader** (Python Generator)-输入train数据的 `batch generator <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/DataLoader_cn.html>`_
- **valid_reader** (Python Generator)-输入valid数据的 `batch generator <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/DataLoader_cn.html>`_
- **place** (fluid.CPUPlace()|fluid.CUDAPlace(N))-该参数表示程序运行在何种设备上，这里的N为GPU对应的ID
- **learning_rate** (float)-模型参数的初始学习率。默认值：0.025。
- **batchsize** (int)-搜索过程数据的批大小。默认值：64。
- **arch_learning_rate** (float)-架构参数的学习率。默认值：3e-4。
- **unrolled** (bool)-是否使用二阶搜索算法。默认值：False。
- **num_epochs** (int)-搜索训练的轮数。默认值：50。
- **epochs_no_archopt** (int)-跳过前若干轮的模型架构参数优化。默认值：0。
- **use_data_parallel** (bool)-是否使用数据并行的多卡训练。默认值：False。
- **log_freq** (int)-每多少步输出一条log。默认值：50。


   .. py:method:: paddleslim.nas.DARTSearch.train()

   对以上定义好的目标网络和数据进行DARTS搜索


**使用示例：**

.. code-block:: python

    import paddle.fluid as fluid
    from paddleslim.nas.darts import DARTSearch

    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model = SuperNet()
        train_reader = batch_generator('train')
        valid_reader = batch_generator('valid')
        searcher = DARTSearch(model, train_reader, valid_reader, place)

        searcher.train()
..

可微分模型架构搜索DARTS
=========

DARTSearch
---------

.. py:class:: paddleslim.nas.DARTSearch(model, train_reader, valid_reader, place, learning_rate=0.025, batchsize=64, num_imgs=50000, arch_learning_rate=3e-4, unrolled=False, num_epochs=50, epochs_no_archopt=0, use_multiprocess=False, use_data_parallel=False, save_dir='./', log_freq=50)

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/release/1.1.0/paddleslim/nas/darts/train_search.py>`_

定义一个DARTS搜索示例，用于在特定数据集和搜索空间上启动模型架构搜索。

**参数：**

- **model** (Paddle Dygraph model)-用于搜索的超网络模型，需要以PaddlePaddle动态图的形式定义。
- **train_reader** (Python Generator)-输入train数据的 `batch generator <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/DataLoader_cn.html>`_
- **valid_reader** (Python Generator)-输入valid数据的 `batch generator <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/DataLoader_cn.html>`_
- **place** (fluid.CPUPlace()|fluid.CUDAPlace(N))-该参数表示程序运行在何种设备上，这里的N为GPU对应的ID
- **learning_rate** (float)-模型参数的初始学习率。默认值：0.025。
- **batchsize** (int)-搜索过程数据的批大小。默认值：64。
- **num_imgs** (int)-数据集总样本数。默认值：50000。
- **arch_learning_rate** (float)-架构参数的学习率。默认值：3e-4。
- **unrolled** (bool)-是否使用二阶搜索算法。默认值：False。
- **num_epochs** (int)-搜索训练的轮数。默认值：50。
- **epochs_no_archopt** (int)-跳过前若干轮的模型架构参数优化。默认值：0。
- **use_multiprocess** (bool)-是否使用多进程的dataloader。默认值：False。
- **use_data_parallel** (bool)-是否使用数据并行的多卡训练。默认值：False。
- **save_dir** (str)-模型参数保存目录。默认值：'./'。
- **log_freq** (int)-每多少步输出一条log。默认值：50。


   .. py:method:: paddleslim.nas.DARTSearch.train()

   对以上定义好的目标网络和数据进行DARTS搜索


**使用示例：**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    from paddleslim.nas.darts import DARTSearch
    
    
    class SuperNet(fluid.dygraph.Layer):
        def __init__(self):
            super(SuperNet, self).__init__()
            self._method = 'DARTS'
            self._steps = 1
            self.stem=fluid.dygraph.nn.Conv2D(
                num_channels=1,
                num_filters=3,
                filter_size=3,
                padding=1)
            self.classifier = fluid.dygraph.nn.Linear(
                input_dim=3072,
                output_dim=10)
            self._multiplier = 4
            self._primitives = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
            self._initialize_alphas()
    
        def _initialize_alphas(self):
            self.alphas_normal = fluid.layers.create_parameter(
                shape=[14, 8],
                dtype="float32")
            self.alphas_reduce = fluid.layers.create_parameter(
                shape=[14, 8],
                dtype="float32")
            self._arch_parameters = [
                self.alphas_normal,
                self.alphas_reduce,
            ]
    
        def arch_parameters(self):
            return self._arch_parameters
    
        def forward(self, input):
            out = self.stem(input) * self.alphas_normal[0][0] * self.alphas_reduce[0][0]
            out = fluid.layers.reshape(out, [0, -1])
            logits = self.classifier(out)
            return logits
    
        def _loss(self, input, label):
            logits = self.forward(input)
            return fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(logits, label))
    
    def batch_generator_creator():
        def __reader__():
            for _ in range(1024):
                batch_image = np.random.random(size=[64, 1, 32, 32]).astype('float32')
                batch_label = np.random.random(size=[64, 1]).astype('int64')
                yield batch_image, batch_label
    
        return __reader__

    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model = SuperNet()
        train_reader = batch_generator_creator()
        valid_reader = batch_generator_creator()
        searcher = DARTSearch(model, train_reader, valid_reader, place)
        searcher.train()

..

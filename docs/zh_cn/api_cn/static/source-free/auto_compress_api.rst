source-free压缩功能
==========

AutoCompression
---------------
.. py:class:: paddleslim.source_free.AutoCompression(model_dir, model_filename, params_filename, save_dir, strategy_config, train_config, train_dataloader, eval_callback, devices='gpu')

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/source_free/auto_compression.py#L32>`_

基于 ``inference mode`` 推理模型使用传入的压缩配置进行模型压缩。

**参数: **

- **model_dir(str)** - 需要压缩的推理模型所在的目录。
- **model_filename(str)** - 需要压缩的推理模型文件名称。
- **params_filename(str)** - 需要压缩的推理模型参数文件名称。
- **save_dir(str)** - 需要压缩后模型的保存目录。
- **strategy_config(dict)** - 使用的压缩策略。字典的关键字必须在: ``QuantizationConfig`` (量化配置), ``DistillationConfig`` (蒸馏配置), 
                     ``MultiTeacherDistillationConfig`` (多teacher蒸馏配置), ``HyperParameterOptimizationConfig`` (超参搜索配置), 
                     ``PruneConfig`` (剪枝配置), ``UnstructurePruneConfig`` (非结构化稀疏配置) 之间选择，目前只支持以下几种配置:
                         1) 量化配置和超参搜索配置 组成 离线量化超参搜索的策略融合;
                         2) 量化配置和蒸馏配置 组成 量化训练和蒸馏的策略融合;
                         3) 剪枝配置和馏配置 组成 结构化剪枝和蒸馏的策略融合;
                         4) 非结构化稀疏配置和馏配置 组成 非结构化稀疏和蒸馏的策略融合;
                         5) 蒸馏配置，单独单蒸馏压缩;
                         6) 多teacher蒸馏配置, 单独多teacher蒸馏配置。
                     每种配置的具体参数信息可以参考：。
- **train_config(dict)** - 训练配置，dict类型。可以配置的参数请参考:  。注意：如果选择离线量化超参搜索的策略融合的话， ``train_config`` 直接设置为 ``None`` 即可。
- **train_dataloader(paddle.io.DataLoader)** - 训练数据迭代器。
- **eval_callback(paddle.io.DataLoader|function)** - eval回调函数，和测试数据之间必须传入一个，如果传入回调函数，则使用回调函数判断模型训练情况。callback传入predict结果（paddle的tensor）。
- **devices(str)** - 确定特定的运行设备，可以是 ``cpu`` , ``gpu``, ``npu``, ``gpu:x``, ``xpu:x``, 或者是 ``npu:x`` 。其中， ``x`` 是GPU, XPU 或者是NPU的编号。当 ``devices`` 是 ``cpu`` 的时候， 程序在CPU上运行， 当 ``devices`` 是 ``gpu:x`` 的时候，程序在GPU上运行， 当device是 ``npu:x`` 的时候，程序在NPU上运行。默认: ``gpu`` 。

**返回：** 一个AutoCompression类的实例。

**示例代码：**

.. code-block:: python

   from paddleslim.source_free import AutoCompression
   ac = AutoCompression()
..
 

.. py:method:: paddleslim.source_free.AutoCompression.compression()

开始进行压缩。


TrainConfig
----------

训练配置。

**参数：**

- **epochs(int)** - 训练的轮数，表明当前数据集需要训练几次。
- **learning_rate(float|LRScheduler)** - 模型优化过程中的学习率。
- **optimizer(str)** - 使用的优化器，需要是 ``paddle.optimizer`` 中优化器的名字, 例如: ``SGD`` 。
- **optim_args(dict)** - 优化器参数。可以指定以下参数：
                        ``grid_clip`` ，指名使用的梯度裁剪的方法，需要是 ``paddle.nn`` 中梯度裁剪的类的名字，例如:  ``ClipGradByValue`` 等。 
                        ``grad_clip_args`` ，梯度裁剪方法中的参数，例如：梯度裁剪选择的方式为 ``ClipGradByValue`` ，那么 ``grad_clip_args`` 可以设置的参数为 ``max`` 和 ``min`` ，参考: `ClipGradByValue <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ClipGradByValue_cn.html#clipgradbyvalue>`_ 。
                        其他优化器中可能需要的参数，例如: ``beta1``, ``beta2``, ``apply_decay_param_fun`` 等，参考: `AdamW <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/AdamW_cn.html#adamw>`_ 。

- **eval_iter(int)** - 训练多少batch的数据进行一次测试。
- **logging_iter(int)** - 训练多少batch的数据进行一次打印。
- **origin_metric(float)** - 要压缩的推理模型的原始精度，可以用来判断实现的eval function是否有问题, 默认： ``None`` 。
- **target_metric(float, optional)** - 如果训练过程中压缩后模型达到了要求的精度，即退出训练，返回当前达到精度的模型，若没有设置该参数，则训练完设置的epochs数量, 默认： ``None`` 。
- **use_fleet(bool, optional)** - 是否使用fleet api去进行分布式训练，默认： ``None`` 。
- **amp_config(dict, optional)** - 如果使用混合精度训练的话，需要配置本参数。参数按照以下规则进行配置：
                                 1) 若不使用fleet api: 
                                     a) 使用 `静态图AMP-O1功能 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/01_paddle2.0_introduction/basic_concept/amp_cn.html#id2>`_ , 需要配置: ``custom_white_list``, ``custom_black_list``, ``custom_black_varnames`` 参数。
          			     b) 使用 `静态图AMP-O2功能 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/01_paddle2.0_introduction/basic_concept/amp_cn.html#id3>`_ , 则需要配置: ``use_pure_fp16`` 和 ``use_fp16_guard`` 参数。
                                 2) 使用fleet api:
                                     参考接口： `amp_config <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#amp_configs>`_ 来进行相对应的参数配置。
- **recompute_config(dict, optional)** - 使用fleet api的前提下可以使用recompute显存优化逻辑。参数按照fleet 接口中所描述的进行配置： `recompute_configs <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#recompute_configs>`_ 。
- **sharding_config(dict, optional)** - 使用fleet api的前提下可以使用sharding 策略。参数按照fleet 接口中所描述的进行配置： `sharding_configs <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#sharding_configs>`_ 。


QuantizationConfig
----------

量化配置。

**参数：**

- **quantize_op_types(list[str])** - 需要进行量化的 op 类型。 
- **weight_bits(int)** - 参数量化bit数。
- **activation_bits(int)** - 激活量化bit数。
- **is_full_quantize(bool)** - 是否量化所有可支持op类型。
- **not_quant_pattern(str|list[str])** - 所有 ``name_scope`` 包含 ``'not_quant_pattern'`` 字符串的 op 都不量化, 设置方式请参考 `fluid.name_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/name_scope_cn.html#name-scope>`_ 。

DistillationConfig
----------

蒸馏配置。

**参数：**

- **distill_loss(str|list[str])** - 蒸馏损失名字，可以设置的损失类型为paddleslim中支持的蒸馏损失，可选的损失函数有: ``fsp_loss``, ``l2_loss``, ``soft_label_loss`` 。如果您需要其他损失函数，可以暂时通过向 `蒸馏损失文件<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py>`_ z中添加相应的损失函数计算，或者通过提issue的方式我们来协助解决。
。
- **distill_node_pair(list[str])** - 蒸馏节点名字列表，每两个节点组成一对，分别属于教师模型和学生模型。
- **distill_lambda(float|list[float])** - 每一个蒸馏损失的权重，长度需要和 ``distill_loss`` 的长度保持一致。
- **teacher_model_dir(str)** - 教师模型的目录。
- **teacher_model_filename(str)** - 教师模型的模型文件名字。
- **teacher_params_filename(str)** - 教师模型的参数文件名字。
- **merge_feed(bool)** - 蒸馏过程是否需要共享同一个输入数据。默认： ``True`` 。


MultiTeacherDistillationConfig
----------

多teacher蒸馏配置。

**参数：**

- **distill_loss(list[str])** - 蒸馏损失名字，可以设置的损失类型为paddleslim中支持的蒸馏损失，可选的损失函数有: ``fsp_loss``, ``l2_loss``, ``soft_label_loss`` 。如果您需要其他损失函数，可以暂时通过向 `蒸馏损失文件<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py>`_ z中添加相应的损失函数计算，或者通过提issue的方式我们来协助解决。
。
- **distill_node_pair(list[list[str]])** - 蒸馏节点名字嵌套列表，教师模型的个数和外部列表的长度需要保持一致。每一个列表代表一个教师模型和学生模型直接的蒸馏节点，其中每两个节点组成一对，分别属于教师模型和学生模型。
- **distill_lambda(list[float])** - 每一个蒸馏损失的权重，长度需要和 ``distill_loss`` 的长度保持一致。
- **teacher_model_dir(list[str])** - 教师模型的目录列表。
- **teacher_model_filename(list[str])** - 教师模型的模型文件名字列表。
- **teacher_params_filename(list[str])** - 教师模型的参数文件名字列表。
- **merge_feed(bool)** - 蒸馏过程是否需要共享同一个输入数据。默认： ``True`` 。


HyperParameterOptimizationConfig
----------

超参搜索搜索空间配置。

.. note::

目前超参搜索仅支持对离线量化算法进行搜索，所以搜索空间配置都是和离线量化相关的配置。

**参数：**

- **ptq_algo(str|list[str])** - 离线量化算法，可为 ``KL``，``mse``, ``'hist``， ``avg``，或者 ``abs_max`` ，该参数仅针对激活值的量化。
- **bias_correct(bool|list[bool])** - 是否使用 bias correction 算法。
- **weight_quantize_type(str|list[str])** - weight的量化方式，可选 ``abs_max`` 或者 ``channel_wise_abs_max`` 。
- **hist_percent(float|list[float])** - ``hist`` 方法的百分位数，设置类型为列表的话，列表中的最大最小值会作为上下界，在上下界范围内进行均匀采样。
- **batch_size(int|list[int])** - 每个batch的图片数量, 设置类型为列表的话，列表中的最大最小值会作为上下界，在上下界范围内进行均匀采样。
- **batch_num(int|list[int])** - 迭代次数, 设置类型为列表的话，列表中的最大最小值会作为上下界，在上下界范围内进行均匀采样。
- **max_quant_count(int)** - 超参搜索运行的最大轮数, 默认：20。

PruneConfig
----------

裁剪配置。

**参数：**

- **prune_algo(str)** - 裁剪算法，可设置为: ``prune`` 或者 ``asp`` 。 ``prune`` 暂时只支持对视觉模型进行压缩， ``asp`` 裁剪暂时只支持对 ``FC`` 进行压缩。
- **pruned_ratio(float)** - 裁剪比例。
- **prune_params_name(list[str])** - 参与裁剪的参数的名字。
- **criterion(str)** - 裁剪算法设置为 ``prune`` 时，评估一个卷积层内通道重要性所参考的指标。目前支持 ``l1_norm``, ``bn_scale``, ``geometry_median`` 。

UnstructurePruneConfig
----------

非结构化稀疏配置。

**参数：**

- **prune_strategy(str, optional)** - 是否使用 ``GMP`` 方式做非结构化稀疏，设置为 ``None`` 的话则不使用 ``GMP`` 进行非结构化稀疏训练，设置为 ``gmp`` 的话则使用 ``GMP`` 进行非结构化稀疏训练。默认：None。
- **prune_mode(str)** - 稀疏化的模式，目前支持的模式有： ``ratio`` 和 ``threshold`` 。在 ``ratio`` 模式下，会给定一个固定比例，例如0.55，然后所有参数中重要性较低的50%会被置0。类似的，在 ``threshold`` 模式下，会给定一个固定阈值，例如1e-2，然后重要性低于1e-2的参数会被置0。
- **threshold(float)** - 稀疏化阈值期望，只有在 ``prune_mode = threshold`` 时才会生效。
- **prune_ratio(float)** - 稀疏化比例期望，只有在 mode== ``ratio`` 时才会生效。
- **gmp_config(dict, optional)** - 使用 ``GMP`` 模式做非结构化稀疏时，需要传入的特殊配置，可以包括以下配置：
                                  ``prune_steps(int)`` - 迭代训练多少iteration后，改变稀疏比例。
                                  ``initial_ratio(float)`` - 初始的稀疏比例。
                                  其它配置可以参考非结构化稀疏接口中 `configs参数 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/prune/unstructured_prune_api.rst#gmpunstrucuturedpruner>`_ 的配置。
- **prune_params_type(str)** - 用以指定哪些类型的参数参与稀疏。目前只支持 ``None`` 和 ``conv1x1_only`` 两个选项，后者表示只稀疏化1x1卷积。而前者表示稀疏化除了归一化的参数。
- **local_sparsity(bool)** - 剪裁比例（ratio）应用的范围： ``local_sparsity`` 开启时意味着每个参与剪裁的参数矩阵稀疏度均为 ``ratio`` ， 关闭时表示只保证模型整体稀疏度达到 ``ratio`` ，但是每个参数矩阵的稀疏度可能存在差异。

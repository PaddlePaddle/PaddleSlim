AutoCompression自动压缩功能
==========

AutoCompression
---------------
.. py:class:: paddleslim.auto_compression.AutoCompression(model_dir, train_dataloader, model_filename, params_filename, save_dir, strategy_config, train_config, eval_callback, devices='gpu')

`源代码 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/auto_compression.py#L49>`_

根据指定的配置对使用 ``paddle.jit.save`` 接口或者 ``paddle.static.save_inference_model`` 接口保存的推理模型进行压缩。

**参数: **

- **model_dir(str)** - 需要压缩的推理模型所在的目录。
- **train_dataloader(paddle.io.DataLoader)** - 训练数据迭代器。注意：如果选择离线量化超参搜索策略的话, ``train_dataloader`` 和 ``eval_dataloader`` 设置相同的数据读取即可。
- **model_filename(str)** - 需要压缩的推理模型文件名称。如果压缩的是onnx模型，则本参数设置为 ``None`` 即可。
- **params_filename(str)** - 需要压缩的推理模型参数文件名称。如果压缩的是onnx模型，则本参数设置为 ``None`` 即可。
- **save_dir(str)** - 压缩后模型的所保存的目录。
- **input_shapes(dict|tuple|list)** - 如果模型除 ``batch size`` 维度外还有可变维度，则需要设置此参数在压缩前固定下来。如果设置的是dict类型，则关键字为输入的名字，对应的值为每个输入的具体shape，例如模型中输入 ``X`` 的形状为 ``[-1, 3, 1, 1]`` 意味着 ``batch size`` 维度、 ``hight`` 维度和 ``width`` 维度都是变化的， ``input_shape`` 可以设置为 ``{"X": [-1, 3, 512, 512]}`` 。如果 ``input_shapes`` 设置为list或者tuple形式的话，模型只能有一个输入，并且输入的形状会设置成 ``input_shapes`` 的形状。设置为 ``None`` 的话，就保持原始形状不变，可能会跳过搜索压缩策略的过程。默认： ``None`` 。
- **train_config(dict)** - 训练配置。可以配置的参数请参考: `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L103>`_ 。注意：如果选择离线量化超参搜索策略的话， ``train_config`` 直接设置为 ``None`` 即可。
- **strategy_config(dict, list(dict), 可选)** - 使用的压缩策略，可以通过设置多个单种策略来并行使用这些压缩方式。字典的关键字必须在: 
             ``QuantAware`` (量化训练配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L55>`_ ), 
             ``QuantPost`` (离线量化配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L187>`_ ), 
             ``Distillation`` (蒸馏配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L107>`_), 
             ``MultiTeacherDistillation`` (多teacher蒸馏配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L134>`_), 
             ``HyperParameterOptimization`` (超参搜索配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L160>`_), 
            ``ChannelPrune`` (结构化稀疏配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L254>`_), 
            ``UnstructurePrune`` (非结构化稀疏配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L288>`_) 之间选择。
            ``ASPPrune`` (ASP半结构化结构化稀疏配置, 可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L268>`_) 之间选择。
            ``TransformerPrune`` (Transformer结构化稀疏配置, 只针对Transformer-encoder结构进行剪枝，可配置的参数参考 `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L278>`_) 之间选择。
            目前关键字只支持以下几种组合策略或者单策略配置:
                         1) ``QuantPost`` & ``HyperParameterOptimization``: 离线量化超参搜索策略;
                         2) ``QuantAware`` & ``Distillation``: 量化训练和蒸馏的策略;
                         3) ``ChannelPrune`` & ``Distillation``: 结构化稀疏和蒸馏的策略;
                         4) ``ASPPrune`` & ``Distillation``: ASP半结构化稀疏和蒸馏的策略;
                         5) ``TransformerPrune`` & ``Distillation``: Transformer结构化稀疏和蒸馏的策略;
                         6) ``UnstructurePrune`` & ``Distillation``: 非结构化稀疏和蒸馏的策略;
                         7) ``Distillation``: 单独单teacher蒸馏策略;
                         8) ``MultiTeacherDistillation``: 多teacher蒸馏策略。
            设置为None的话会自动的选择策略去做压缩。默认：None。
- **target_speedup(float, 可选)** - 目标加速比例，在支持硬件延时表的设备上会根据预估的加速进行压缩策略选择；在硬件延时表不支持的设备上会默认量化相比 ``float32`` 加速70%，剩下的加速比会等价设置成剪枝的比例（压缩后模型实测的加速情况和预计差别可能较大，暂时不太推荐在硬件延时表不支持的设备上使用本参数）。默认： ``None`` 。
- **eval_callback(function, 可选)** - eval回调函数，使用回调函数判断模型训练情况, 回调函数的写法参考： `<//github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/auto-compression/custom_function.rst>`_ 。 ``eval_callback`` 和 ``eval_dataloader`` 不能都设置为None。默认：None。
- **eval_dataloader(paddle.io.Dataloader, 可选)** - 如果传入测试数据迭代器，则使用 ``EMD`` 距离判断压缩前后模型之间的差别，目前仅支持离线量化超参搜索使用这种方式判断压缩前后模型的压缩。
- **deploy_hardware(str, 可选)** - 压缩后模型的部署硬件。默认： ``gpu`` 。

**返回：** 一个AutoCompression类的实例。

**示例代码：**

```shell

   import paddle

   from paddleslim.auto_compression import AutoCompression

   default_qat_config = {

       "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],

       "weight_bits": 8,

       "activation_bits": 8,

       "is_full_quantize": False,

       "not_quant_pattern": ["skip_quant"],

   }

   default_distill_config = {

       "loss": args.loss,

       "node": args.node,

       "alpha": args.alpha,

       "teacher_model_dir": args.teacher_model_dir,

       "teacher_model_filename": args.teacher_model_filename,

       "teacher_params_filename": args.teacher_params_filename,

   }

   train_dataloader = Cifar10(mode='train')

   eval_dataloader = Cifar10(mode='eval')

   ac = AutoCompression(model_path, train_dataloader, model_filename, params_filename, save_dir, \

                        strategy_config="QuantPost": QuantPost(**default_ptq_config), 

                        "Distillation": HyperParameterOptimization(**default_distill_config)}, \

                        train_config=None, eval_dataloader=eval_dataloader,devices='gpu')

```
 

.. py:method:: paddleslim.auto_compression.AutoCompression.compress()

开始进行压缩。


TrainConfig
----------

训练超参配置。

**参数：**

- **epochs(int)** - 训练的轮数，表明当前数据集需要训练几次。
- **train_iter(int, optional)** 训练的迭代次数，表明需要迭代多少批次的数据，和 ``epoch`` 之间仅需要设置一个。
- **learning_rate(float|dict)** - 模型优化过程中的学习率, 如果是dict类型，则dict的关键字如下： ``type``: 学习率策略的类名，可参考 ``paddle.optimizer.lr`` 中的类设置,
                                  其它关键字根据实际调用的学习率的策略中的参数设置。
- **optimizer_builder(dict)** - 使用的优化器和相关配置。dict中对应的关键字如下：
                        ``optimizer(dict)``: 指定关键字 ``type`` 需要是 ``paddle.optimizer`` 中优化器的类名, 例如: ``SGD`` ，其他关键字根据具体使用的优化器中的参数设置。
                        ``weight_decay(float, optional)``: 压缩训练过程中的参数衰退。
                        ``regularizer(dict)``: 指定关键字 ``type`` 需要是 ``paddle.regularizer`` 中的权重衰减正则类名，其他关键字根据具体使用的类中的参数设置。
                        ``grid_clip`` ，指名使用的梯度裁剪的方法，需要是 ``paddle.nn`` 中梯度裁剪的类的名字，例如:  ``ClipGradByValue`` 等，其他关键字根据具体使用的类中的参数设置。 

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
- **sparse_model(bool, optional)** - 设置 ``sparse_model`` 为 True, 可以移出非结构化稀疏产出的模型中多余的mask tensor的变量，默认: False.

QuantAware
----------

量化训练配置。

**参数：**

- **use_pact(bool)** - 是否开启PACT。一般情况下，开启PACT后，量化产出的模型精度会更高。算法原理请参考： `PACT: Parameterized Clipping Activation for Quantized Neural Networks <https://arxiv.org/abs/1805.06085>`_
- **weight_quantize_type(str)** - 参数量化方式，可选: ['channel_wise_abs_max', 'abs_max', 'moving_average_abs_max', 'range_abs_max']。如果使用 TensorRT 加载量化后的模型来预测，请使用 'channel_wise_abs_max' 。 默认 'channel_wise_abs_max' 。
- **quantize_op_types(list[str])** - 需要进行量化的 op 类型。通过以下代码输出所有支持量化的OP类型：
```python
from paddleslim.quant.quanter import TRANSFORM_PASS_OP_TYPES,QUANT_DEQUANT_PASS_OP_TYPES
print(TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES)
```
- **onnx_format(bool)** - 是否采用ONNX量化格式标准，如果需要导出成ONNX，则需要设置为True。默认：False。
- **weight_bits(int)** - 参数量化bit数。默认：8.
- **activation_bits(int)** - 激活量化bit数。默认：8。
- **activation_quantize_type(str)** - 激活量化方式，可选 'abs_max' , 'range_abs_max' , 'moving_average_abs_max' 。如果使用 TensorRT 加载量化后的模型来预测，请使用 'range_abs_max' 或 'moving_average_abs_max' 。默认为 'moving_average_abs_max'。
- **not_quant_pattern(str|list[str])** - 所有 ``name_scope`` 包含 ``'not_quant_pattern'`` 字符串的 op 都不量化, 设置方式请参考 `fluid.name_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/name_scope_cn.html#name-scope>`_ 。默认：'skip_quant'.
- **window_size(int)** - 'range_abs_max' 量化方式的 window size ，默认10000。
- **moving_rate(float)** - 'moving_average_abs_max' 量化方式的衰减系数，默认 0.9。
- **for_tensorrt(bool)** - 量化后的模型是否使用 TensorRT 进行预测。默认值为False. 通过以下代码，输出for_tensorrt=True时会量化到的OP：
```python
from paddleslim.quant.quanter import TENSORRT_OP_TYPES
print(TENSORRT_OP_TYPES)
```
- **is_full_quantize(bool)** - 是否量化所有可支持op类型。默认：False。

Distillation
----------

蒸馏配置。

**参数：**

- **loss(str|list[str])** - 蒸馏损失名字，可以设置的损失类型为paddleslim中支持的蒸馏损失，可选的损失函数有: ``fsp``, ``l2``, ``soft_label`` 。如果您需要其他损失函数，可以暂时通过向 `蒸馏损失文件<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py>`_ z中添加相应的损失函数计算，或者通过提issue的方式我们来协助解决。
。
- **node(list[str])** - 蒸馏节点名字列表，可以选择：1. 使用自蒸馏的话，蒸馏结点仅包含学生网络节点即可, 支持多节点蒸馏; 2. 使用其他蒸馏的话，蒸馏节点需要包含教师网络节点和对应的学生网络节点, 每两个节点组成一对，分别属于教师模型和学生模型。
- **alpha(float|list[float])** - 每一个蒸馏损失的权重，长度需要和 ``loss`` 的长度保持一致。
- **teacher_model_dir(str)** - 教师模型的目录。
- **teacher_model_filename(str)** - 教师模型的模型文件名字。
- **teacher_params_filename(str)** - 教师模型的参数文件名字。


MultiTeacherDistillation
----------

多teacher蒸馏配置。

**参数：**

- **loss(list[str])** - 蒸馏损失名字，可以设置的损失类型为paddleslim中支持的蒸馏损失，可选的损失函数有: ``fsp``, ``l2``, ``soft_label`` 。如果您需要其他损失函数，可以暂时通过向 `蒸馏损失文件<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py>`_ 中添加相应的损失函数计算，或者通过提issue的方式我们来协助解决。
。
- **node(list[list[str]])** - 蒸馏节点名字嵌套列表，教师模型的个数和外部列表的长度需要保持一致。每一个列表代表一个教师模型和学生模型直接的蒸馏节点，其中每两个节点组成一对，分别属于教师模型和学生模型。
- **alpha(list[float])** - 每一个蒸馏损失的权重，长度需要和 ``distill_loss`` 的长度保持一致。
- **teacher_model_dir(list[str])** - 教师模型的目录列表。
- **teacher_model_filename(list[str])** - 教师模型的模型文件名字列表。
- **teacher_params_filename(list[str])** - 教师模型的参数文件名字列表。


HyperParameterOptimization
----------

超参搜索搜索空间配置。

.. note::

目前超参搜索仅支持对离线量化算法进行搜索，所以搜索空间配置都是和离线量化相关的配置。

**参数：**

- **ptq_algo(str|list[str])** - 离线量化算法，可为 ``KL``，``mse``, ``'hist``， ``avg``，或者 ``abs_max`` ，该参数仅针对激活值的量化。
- **bias_correct(bool|list[bool])** - 是否使用 bias correction 算法。
- **weight_quantize_type(str|list[str])** - weight的量化方式，可选 ``abs_max`` 或者 ``channel_wise_abs_max`` 。
- **hist_percent(float|list[float])** - ``hist`` 方法的百分位数，设置类型为列表的话，列表中的最大最小值会作为上下界，在上下界范围内进行均匀采样。
- **batch_num(int|list[int])** - 迭代次数, 设置类型为列表的话，列表中的最大最小值会作为上下界，在上下界范围内进行均匀采样。
- **max_quant_count(int)** - 超参搜索运行的最大轮数, 默认：20。

ChannelPrune
----------

结构化稀疏配置。

**参数：**

- **pruned_ratio(float)** - 每个卷积层的通道数被剪裁的比例。
- **prune_params_name(list[str])** - 参与裁剪的参数的名字。如果设置为 ``None`` , 则会按照传入的剪枝比例对所有可以裁剪的卷积层进行裁剪。合适的卷积层可以通过计算每一层的敏感度来选择，敏感度可以通过 `敏感度计算工具 <../../../../../example/auto_compression/prune_sensitivity_analysis/>`_ 来获得每层的敏感度信息，然后设置合适的裁剪的卷积层名字。也可以使用 `Netron工具 <https://netron.app/`_  可视化`*.pdmodel`模型文件，选择合适的卷积层进行剪裁。默认： ``None`` 。

- **criterion(str)** - 评估一个卷积层内通道重要性所参考的指标。目前支持 ``l1_norm``, ``bn_scale``, ``geometry_median`` 。具体定义和使用可参考 `结构化稀疏API文档 <https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html`_ 。

ASPPrune
----------

ASP半结构化稀疏配置

**参数：**

- **prune_params_name(list[str])** - 待剪裁的卷积层的权重名称。如果设置为 ``None``, 则会按照传入的剪枝比例对所有可以裁剪的卷积层进行裁剪。或者，使用 `Netron工具 <https://netron.app/>`_ 可视化`*.pdmodel`模型文件，选择合适的卷积层进行剪裁。默认： ``None`` 。

TransformerPrune
----------

针对Transformer结构的结构化剪枝参数

- **pruned_ratio(float)** - 每个全链接层的被剪裁的比例。

UnstructurePrune
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
- **prune_params_type(str)** - 用以指定哪些类型的参数参与稀疏。目前只支持 ``None`` 和 ``conv1x1_only`` 两个选项，后者表示只稀疏化1x1卷积。而前者表示稀疏化除了归一化的参数。默认： ``conv1x1_only`` 。
- **local_sparsity(bool)** - 剪裁比例（ratio）应用的范围： ``local_sparsity`` 开启时意味着每个参与剪裁的参数矩阵稀疏度均为 ``ratio`` ， 关闭时表示只保证模型整体稀疏度达到 ``ratio`` ，但是每个参数矩阵的稀疏度可能存在差异。

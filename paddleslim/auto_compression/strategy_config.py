#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple

__all__ = [
    "BaseStrategy",
    "QuantAware",
    "Distillation",
    "MultiTeacherDistillation",
    "HyperParameterOptimization",
    "ChannelPrune",
    "UnstructurePrune",
    "TransformerPrune",
    "ASPPrune",
    "merge_config",
    "ProgramInfo",
    "TrainConfig",
    "SUPPORTED_CONFIG",
    "TRAIN_CONFIG_NAME",
    "QuantPost",
]

SUPPORTED_CONFIG = [
    "QuantAware",
    "Distillation",
    "MultiTeacherDistillation",
    "HyperParameterOptimization",
    "ChannelPrune",
    "UnstructurePrune",
    "TransformerPrune",
    "ASPPrune",
    "QuantPost",
]

TRAIN_CONFIG_NAME = "TrainConfig"


class BaseStrategy:
    def __init__(self, name):
        self.name = name


class QuantAware(BaseStrategy):
    def __init__(self,
                 quantize_op_types=[
                     'conv2d', 'depthwise_conv2d', 'conv2d_transpose', 'mul',
                     'matmul', 'matmul_v2'
                 ],
                 weight_bits=8,
                 activation_bits=8,
                 not_quant_pattern=['skip_quant'],
                 use_pact=False,
                 activation_quantize_type='moving_average_abs_max',
                 weight_quantize_type='channel_wise_abs_max',
                 dtype='int8',
                 window_size=10000,
                 moving_rate=0.9,
                 for_tensorrt=False,
                 onnx_format=True,
                 is_full_quantize=False):
        """
        Quantization Config.
        Args:
            quantize_op_types(list(str)): Ops of type in quantize_op_types, will be quantized. Default: ['conv2d', 'depthwise_conv2d', 'mul', 'matmul', 'matmul_v2'].
            weight_bits(int): Weight quantize bit num. Default: 8.
            activation_bits(int): Activation quantize bit num. Default 8.
            not_quant_pattern(list(str)): Ops of name_scope in not_quant_pattern list, will not be quantized. Default: 'skip_quant'.
            use_pact(bool): Whether to use pact in quantization training. Default: False.
            activation_quantize_type(str): Activation quantize type. Default is 'moving_average_abs_max'.
            weight_quantize_type(str): Weight quantize type. Default 'channel_wise_abs_max'.
            dtype(str): Data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'.
            window_size(int): Window size for 'range_abs_max' quantization. Default: 10000.
            moving_rate(float): The decay coefficient of moving average. Default: 0.9.
            for_tensorrt(bool): If True, 'quantize_op_types' will be TENSORRT_OP_TYPES. Default: False.
            onnx_format(bool): Whether to export the quantized model with format of ONNX. Default is False.
            is_full_quantize(bool): If True, 'quantoze_op_types' will be TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES. Default: False.
        """
        super(QuantAware, self).__init__("QuantAware")
        self.quantize_op_types = quantize_op_types
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.not_quant_pattern = not_quant_pattern
        self.use_pact = use_pact
        self.is_full_quantize = is_full_quantize
        self.activation_quantize_type = activation_quantize_type
        self.weight_quantize_type = weight_quantize_type
        self.dtype = dtype
        self.window_size = window_size
        self.moving_rate = moving_rate
        self.for_tensorrt = for_tensorrt
        self.onnx_format = onnx_format
        self.is_full_quantize = is_full_quantize


class Distillation(BaseStrategy):
    def __init__(self,
                 loss='l2',
                 node=[],
                 alpha=1.0,
                 teacher_model_dir=None,
                 teacher_model_filename=None,
                 teacher_params_filename=None):
        """
        Distillation Config.
        Args:
            loss(str|list(str)): Distillation loss, the type of loss can be set reference `<https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/dist/single_distiller_api.html>`_. If set list of loss, means the difference node can be set difference distill loss, the length of loss must equal to length of node. Default: 'l2'.
            node(list(str)|list(list(str))): Distillation node, users can set node from the model before compress. If set list of list, every inner list used same distill loss, the length of list must equal to length of loss.  Default: [].
            alpha(float|list(float)): The lambda of distillation loss. If set list of alpha, the length of alpha must equal to length of loss. Default: 1.0. 
            teacher_model_dir(str, optional): The path of teacher inference model, and the model and params that saved by ``paddle.static.io.save_inference_model`` are under the path. If set to None, the teacher model will be set to the model before compress. Default: None.
            teacher_model_filename(str, optional): The name of teacher model file. If parameters are saved in separate files, set it as 'None'. Default: 'None'.
            teacher_params_filename(str, optional): The name of teacher params file. When all parameters are saved in a single file, set it as filename. If parameters are saved in separate files, set it as 'None'. Default : 'None'.
        """
        super(Distillation, self).__init__("Distillation")
        self.loss = loss
        self.node = node
        self.alpha = alpha
        self.teacher_model_dir = teacher_model_dir
        self.teacher_model_filename = teacher_model_filename
        self.teacher_params_filename = teacher_params_filename


class MultiTeacherDistillation:
    def __init__(self,
                 loss=[],
                 node=[],
                 alpha=[],
                 teacher_model_dir=[],
                 teacher_model_filename=[],
                 teacher_params_filename=[]):
        """
        Multi-Teacher Distillation Config.
        Args:
            loss(list(str)): The list of distillation loss, the type of loss can be set reference `<https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/dist/single_distiller_api.html>`_. One-to-one correspondence between loss and teacher model. Default: [].
            node(list(list(str))): Distillation node, users can set node from the model before compress. If set list of list, every inner list used same distill loss, the length of list must equal to length of loss.  Default: [].
            alpha(list(float)): The list of lambda of distillation loss. One-to-one correspondence between alpha and loss. Default: []. 
            teacher_model_dir(list): The list of path of teacher inference model, and the model and params that saved by ``paddle.static.io.save_inference_model`` are under the path. If set to None, the teacher model will be set to the model before compress. Default: None.
            teacher_model_filename(list): The list of name of teacher model file. If parameters are saved in separate files, set it as 'None'. Default: 'None'.
            teacher_params_filename(list): The list of name of teacher params fie. When all parameters are saved in a single file, set it as filename. If parameters are saved in separate files, set it as 'None'. Default : 'None'.
        """
        self.loss = loss
        self.node = node
        self.alpha = alpha
        self.teacher_model_dir = teacher_model_dir
        self.teacher_model_filename = teacher_model_filename
        self.teacher_params_filename = teacher_params_filename


class HyperParameterOptimization(BaseStrategy):
    def __init__(self,
                 ptq_algo=["KL", "hist", "avg", "mse"],
                 bias_correct=[True, False],
                 weight_quantize_type=['channel_wise_abs_max'],
                 hist_percent=[0.98, 0.999],
                 batch_num=[10, 30],
                 max_quant_count=20):
        """
        HyperParameterOptimization Config.
        Args:
            ptq_algo(list(str)): Post-Training Quantization algorithm, can be set reference the algo from `<https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/quant/quantization_api.html#quant-post-static>`_.
            bias_correct(list(bool)): Whether to use bias_correct.
            weight_quantize_type(list(str)): Quantization type for weight, can be set from 'channel_abs_max' or 'abs_max'.
            hist_percent(list(float)): The upper and lower bounds of threshold of algo 'hist' for activations, the real percent is uniform sampling in this bounds.
            batch_num(list(int)): The upper and lower bounds of batch number, the real batch number is uniform sampling in this bounds.
            max_quant_count(int): Max number of model quantization. Default: 20.
        """
        super(HyperParameterOptimization, self).__init__("HPO_PTQ")
        self.ptq_algo = ptq_algo
        self.bias_correct = bias_correct
        self.weight_quantize_type = weight_quantize_type
        self.hist_percent = hist_percent
        self.batch_num = batch_num
        self.max_quant_count = max_quant_count


class QuantPost(BaseStrategy):
    def __init__(self,
                 batch_size=32,
                 batch_nums=None,
                 epochs=20,
                 lr=0.1,
                 algo='hist',
                 hist_percent=0.999,
                 regions=None,
                 region_weights_names=None,
                 recon_level=None,
                 is_full_quantize=False,
                 bias_correction=False,
                 weight_quantize_type='channel_wise_abs_max',
                 activation_quantize_type='range_abs_max',
                 simulate_activation_quant=False,
                 skip_tensor_list=None,
                 onnx_format=True,
                 quantize_op_types=[
                     "conv2d", "depthwise_conv2d", "mul", "matmul", "matmul_v2"
                 ],
                 weight_bits=8,
                 activation_bits=8):
        """
        QuantPost Config.
        Args:
            batch_size(int, optional): The batch size of DataLoader. Default: 1.
            batch_nums(int, optional): If batch_nums is not None, the number of calibrate data is 'batch_size*batch_nums'. If batch_nums is None, use all data generated by sample_generator as calibrate data. Default: None.
            lr(float, optional): The learning rate of Reconstruction Quanter. Default: 0.1.
            algo(str, optional): Post-Training Quantization algorithm, can be set reference the algo from `<https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/quant/quantization_api.html#quant-post-static>`. Default: 'hist'.
            hist_percent(float, optional): The percentile of histogram for algo hist. Default: 0.999.
            regions(list[list], optional): The list of some regions, each region is a subgraph of fp32 program and it will have exact 1 input operation and 1 output operation. When the recon-level is region, the reconstruction loss of each region is minimized. Default: None.
            region_weights_names(list[list], optional): The weight names inside every region. Default: None.
            recon_level(str, optional): The type of reconstruction granularity. Currently support ['layer-wise', 'region-wise'] types. Only when recon_level isn't None can Reconstruction Quanter be used. Default: None. 
            is_full_quantize(bool): If True, 'quantoze_op_types' will be TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES. Default: False.
            bias_correct(list(bool)): Whether to use bias correction method of https://arxiv.org/abs/1810.05723. Default: False.
            weight_quantize_type(str): Weight quantize type. Default: 'channel_wise_abs_max'.
            activation_quantize_type(str): Activation quantize type. Default: 'moving_average_abs_max'.
            simulate_activation_quant(bool, optional): Whether we need the noise caused by activation quantization during the reconstruction process. Default: False.
            skip_tensor_list(list): List of skip quant tensor name. Default: None.
            onnx_format(bool): Whether to export the quantized model with format of ONNX. Default: False.
            quantize_op_types(list(str)): Ops of type in quantize_op_types, will be quantized. Default: ['conv2d', 'depthwise_conv2d', 'mul', 'matmul', 'matmul_v2'].
            weight_bits(int): Weight quantize bit num. Default: 8.
            activation_bits(int): Activation quantize bit num. Default: 8.
        """
        super(QuantPost, self).__init__("PTQ")
        self.batch_size = batch_size
        self.batch_nums = batch_nums
        self.epochs = epochs
        self.lr = lr
        self.algo = algo
        self.hist_percent = hist_percent
        self.regions = regions
        self.region_weights_names = region_weights_names
        self.recon_level = recon_level
        self.is_full_quantize = is_full_quantize
        self.bias_correction = bias_correction
        self.weight_quantize_type = weight_quantize_type
        self.activation_quantize_type = activation_quantize_type
        self.simulate_activation_quant = simulate_activation_quant
        self.skip_tensor_list = skip_tensor_list
        self.onnx_format = onnx_format
        self.quantize_op_types = quantize_op_types
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits


class ChannelPrune:
    def __init__(self,
                 pruned_ratio,
                 prune_params_name=None,
                 criterion='l1_norm'):
        """
        ChannelPrune Config.
        Args:
            pruned_ratio(float|list[float]): The ratios to be pruned.
            prune_params_name(list(str)): A list of parameter names to be pruned.
            criterion(str|function): the criterion used to sort channels for pruning, can be choose from ['l1_norm', 'bn_scale', 'geometry_median']. Default: 'l1_norm'.
        """
        self.pruned_ratio = pruned_ratio
        self.prune_params_name = prune_params_name
        self.criterion = criterion


class ASPPrune:
    def __init__(self, prune_params_name=None):
        """
        ASPPrune Config.
        Args:
            prune_params_name(list(str)): A list of parameter names to be pruned.
        """
        self.prune_params_name = prune_params_name


class TransformerPrune:
    def __init__(self, pruned_ratio):
        """
        TransformerPrune Config.
        Args:
            pruned_ratio(float): The ratios to be pruned each fully-connected layer.
        """
        self.pruned_ratio = pruned_ratio


class UnstructurePrune:
    def __init__(self,
                 prune_strategy=None,
                 prune_mode='ratio',
                 threshold=0.01,
                 ratio=0.55,
                 gmp_config=None,
                 prune_params_type='conv1x1_only',
                 local_sparsity=False):
        """
        UnstructurePrune Config.
        Args:
            prune_strategy(str, optional): The pruning strategy, currently we support base and gmp, ``None`` means use base pruning strategy. Default: ``None``.
            prune_mode(str): The pruning mode: whether by ratio or by threshold. Default: 'ratio'.
            threshold(float): The threshold to set zeros, the abs(weights) lower than which will be zeros. Default: 0.01.
            ratio(float): The ratio to set zeros, the smaller portion will be zeros. Default: 0.55.
            gmp_config(dict): The dictionary contains all the configs for GMP pruner. Default: None. The detailed description is as below:
              .. code-block:: python
                     
                     {'stable_iterations': int} # the duration of stable phase in terms of global iterations
                     {'pruning_iterations': int} # the duration of pruning phase in terms of global iterations
                     {'tunning_iterations': int} # the duration of tunning phase in terms of global iterations
                     {'resume_iteration': int} # the start timestamp you want to train from, in terms if global iteration
                     {'pruning_steps': int} # the total times you want to increase the ratio
                     {'initial_ratio': float} # the initial ratio value
              
              ..
            prune_params_type(str): Which kind of params should be pruned, we only support None (all but norms) and conv1x1_only for now. Default: None.
            local_sparsity(bool): Whether to prune all the parameter matrix at the same ratio or not. Default: False.
        """
        self.prune_strategy = prune_strategy
        self.prune_mode = prune_mode
        self.threshold = threshold
        self.ratio = ratio
        self.gmp_config = gmp_config
        self.prune_params_type = prune_params_type
        self.local_sparsity = local_sparsity


class TrainConfig:
    def __init__(self,
                 epochs=None,
                 train_iter=None,
                 learning_rate=0.02,
                 optimizer_builder={'optimizer': {
                     'type': 'SGD'
                 }},
                 eval_iter=1000,
                 logging_iter=10,
                 origin_metric=None,
                 target_metric=None,
                 amp_config=None,
                 recompute_config=None,
                 sharding_config=None,
                 sparse_model=False):
        """
        Train Config.
        Args:
            epochs(int): The number of total epochs. Default: None.
            train_iter(int):  Training total iteration, `epochs` or `train_iter` only need to set one. Default: None.
            learning_rate(float|dict): learning rate in the training. If set dict, the detailed description of learning_rate is as blow: 
              .. code-block:: python
                     
                  'type'(str) # the class name of learning rate decay, can reference in paddle.optimizer.lr.
              ..
              other keys in the learning_rate depend on the parameters in the class of learning rate decay. 
              Such as, if you want to use ``PiecewiseDecay``, need to set learning_rate like: 
              {'type': PiecewiseDecay, 'boundaries': [4500], 'values': [0.005, 0.0005]}.
            optimizer_builder(str|dict): optimizer in th training. If set dict, the detailed description of optimizer_builder is as blow:
              .. code-block:: python
                     
                  'optimizer'(dict) # the 'type' in the optimizer need to be the class name in the paddle.optimizer,  
                                      other key of optimzer depend on the parameters in the class.
                  'weight_decay(float, optional)' # weight decay in the training.
                  'regularizer(dict)': # the 'type' in the regularizer need to be the class name in the paddle.regularizer, 
                                         other key of optimzer depend on the parameters in the class.
                  'grad_clip(dict)': # the 'type' in the grad_clip need to be the class name in the paddle.nn, such as: 'ClipGradByGlobalNorm',
                                     other key of grad_clip depend on the parameters in the class.
              ..
            eval_iter(int): Test period in batches. Default: 1000.
            logging_iter(int): Log period in batches. Default: 10.
            origin_metric(float, optional): The Metric of model before compress, used to check whether the dataloader is correct if is not None. Default: None.
            target_metric(float, optional): The Metric of model after compress, if set target metric, the metric of compressed model satisfy the requirements, will be stop training. If not set, will train epochs as users set. Default: None.
            amp_config(dict, optional): The dictionary contains all the configs of amp. Default: None. The detailed description is as below when turning on distributed training: 
              .. code-block:: python
                 AMP-O1 `<https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/amp_cn.html#id2>`_ : 
                     {'custom_white_list', set} # The custom white_list. It's the set of ops that support
                         fp16 calculation and are considered numerically-safe and performance-critical. These ops 
                         will be converted to fp16.
                     {'custom_black_list': set} # The custom black_list. The set of ops that support fp16
                         calculation and are considered numerically-dangerous and whose effects may also be 
                         observed in downstream ops. These ops will not be converted to fp16.
                     {'custom_black_varnames': set} # Users' custom black varibles' names.

                 AMP-O2 `<https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/amp_cn.html#id3>`_ : 
                     {'use_pure_fp16': bool} # Whether to use the pure fp16 training.
                     {'use_fp16_guard': bool} # Whether to use `fp16_guard` when constructing the program.
              ..
              If you want to use AMP-O2, you need to set use_pure_fp16 is True and use_fp16_guard is False.
              when turning on distributed training, the key of amp_config can reference `<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#amp-configs>`_.

            recompute_config(dict, optional): The dictionary contains all the configs of recompute. Default: None. The recompute config only can be set when turning on distributed training, the key of recompute_config can reference `<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#recompute-configs>`_. 
            sharding_config(dict, optional): The dictionary contains all the configs of sharding. Default: None. The sharding config only can be set when turning on distributed training, the key of sharding_config can reference `<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#sharding-configs>`_.
            sparse_model(bool, optional): Set sparse_model to ``True`` to remove mask tensor when the compress strategy is unstructure prune. Default: False.
        """
        self.epochs = epochs
        self.train_iter = train_iter
        self.learning_rate = learning_rate
        self.optimizer_builder = optimizer_builder
        self.eval_iter = eval_iter
        self.logging_iter = logging_iter
        self.origin_metric = origin_metric
        self.target_metric = target_metric
        self.amp_config = amp_config
        self.recompute_config = recompute_config
        self.sharding_config = sharding_config
        self.sparse_model = sparse_model


class MergeConfig:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


def merge_config(*args):
    fields = set()
    cfg = dict()
    for arg in args:
        cfg.update(arg.__dict__)
    return MergeConfig(**cfg)


class ProgramInfo:
    def __init__(self,
                 startup_program,
                 program,
                 feed_target_names,
                 fetch_targets,
                 optimizer=None,
                 learning_rate=None,
                 loss_dict=None):
        """
        ProgramInfo Config.
        Args:
            startup_program(paddle.static.Program): Startup program, the means of startup program can reference `<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/default_startup_program_cn.html#default-startup-program>`_.
            program(paddle.static.Program): main program, the means of main program can reference `<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/default_main_program_cn.html#default-main-program>`_.
            feed_target_names(list(str)): The name of feed tensor in the program.
            fetch_targets(list(Variable)): The fetch variable in the program.
            optimizer(Optimizer, optional): Optimizer in training. Default: None.
            learning_rate(float|paddle.optimizer.lr, optional): learning_rate in training. Default: None.
            loss_dict(dict): The components of losses.
        """
        self.startup_program = startup_program
        self.program = program
        self.feed_target_names = feed_target_names
        self.fetch_targets = fetch_targets
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_dict = loss_dict

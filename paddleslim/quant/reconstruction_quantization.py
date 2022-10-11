#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
import math
import shutil
import logging
import numpy as np
import copy
import time
import sys
import paddle
import paddle.fluid as fluid
import six
try:
    from tqdm import tqdm
except:
    from paddle.fluid.contrib.slim.quantization.utils import tqdm
from inspect import isgeneratorfunction
from paddle.fluid import io
from paddle.fluid import core
from paddle.fluid import reader
from paddle.fluid import framework
from paddle.fluid import unique_name
from paddle.fluid.executor import global_scope, Executor
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass, QuantizationTransformPassV2, QuantizationFreezePass, QuantWeightPass, AddQuantDequantPass, AddQuantDequantPassV2
from paddle.fluid.contrib.slim.quantization import cal_kl_threshold
from paddle.fluid.contrib.slim.quantization import utils
from ..dist import merge
from ..core.graph_wrapper import GraphWrapper
from ..common import get_logger

__all__ = [
    'ReconstructionQuantization',
]

_logger = get_logger(__name__,
                     logging.INFO,
                     fmt='%(asctime)s-%(levelname)s: %(message)s')

GAMMA = -0.1
ZETA = 1.1
def _all_persistable_var_names(program):
    persistable_var_names = []
    for var in program.list_vars():
        if var.persistable:
            persistable_var_names.append(var.name)
    return persistable_var_names


def _remove_unused_var_nodes(graph):
    all_used_vars = set()
    ops = graph.all_op_nodes()
    for op_node in ops:
        for input_node in op_node.inputs:
            all_used_vars.add(input_node)
        for output_node in op_node.outputs:
            all_used_vars.add(output_node)

    all_used_vars = {n.node for n in all_used_vars}
    all_unused_vars = {
        n
        for n in filter(lambda node: node.node not in all_used_vars,
                        graph.all_var_nodes())
    }
    graph.safe_remove_nodes(all_unused_vars)
    return graph


def _remove_ctrl_vars(graph):
    remove_ctr_vars = set()
    for node in graph.all_var_nodes():
        if node.is_ctrl_var():
            remove_ctr_vars.add(node)
    graph.safe_remove_nodes(remove_ctr_vars)
    return graph


def _apply_pass(scope,
                graph,
                pass_name,
                attrs=None,
                attr_values=None,
                debug=False):
    ir_pass = core.get_pass(pass_name)
    cpp_graph = graph.graph
    if not cpp_graph.has('__param_scope__'):
        cpp_graph.set_not_owned('__param_scope__', scope)
    if attrs:
        assert attr_values and len(attrs) == len(
            attr_values
        ), "Different number of pass attributes and their values."
        for attr, value in zip(attrs, attr_values):
            ir_pass.set(attr, value)
    ir_pass.apply(cpp_graph)
    if debug:
        graph.draw('.', 'qat_fp32_{}'.format(pass_name), graph.all_op_nodes())
    _remove_unused_var_nodes(graph)
    return graph


class ReconstructionQuantization(object):
    """
    Utilizing reconstruction quantization method to quantize the FP32 model,
    and it uses calibrate data to get the quantization information for all
    quantized variables.
    """

    def __init__(self,
                 executor,
                 model_dir,
                 scope=None,
                 model_filename=None,
                 params_filename=None,
                 batch_generator=None,
                 sample_generator=None,
                 data_loader=None,
                 batch_size=10,
                 batch_nums=None,
                 algo="KL",
                 hist_percent=0.99999,
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 recon_grit='layer-wise',
                 is_intro_act_noise=False,
                 learning_rate=0.1,
                 is_full_quantize=False,
                 bias_correction=False,
                 activation_bits=8,
                 weight_bits=8,
                 activation_quantize_type='range_abs_max',
                 weight_quantize_type='channel_wise_abs_max',
                 onnx_format=False,
                 freeze_model=True,
                 optimize_model=False,
                 is_use_cache_file=False,
                 skip_tensor_list=None,
                 same_scale_tensor_list=None,
                 cache_dir=None,
                 scale_dict=None,
                 return_graph=False,
                 blocks=None,
                 block_weights_names=None,
                 epochs=20,
                 scale_trainable=False):
        '''
        Constructor.

        Args:
            executor(fluid.Executor): The executor to load, run and save the
                quantized model.
            scope(fluid.Scope, optional): The scope of the program, use it to load
                and save variables. If scope=None, get scope by global_scope().
            model_dir(str): The path of the fp32 model that will be quantized,
                and the model and params files are under the path.
            model_filename(str, optional): The name of file to load the inference
                program. If it is None, the default filename '__model__' will
                be used. Default is 'None'.
            params_filename(str, optional): The name of file to load all parameters.
                When all parameters were saved in a single binary file, set it
                as the real filename. If parameters were saved in separate files,
                set it as 'None'. Default is 'None'.
            batch_generator(Python Generator): The batch generator provides
                calibrate data for DataLoader, and it returns a batch every
                time. Note that, sample_generator and batch_generator, only one
                should be set. Beisdes, batch_generator supports lod tensor.
            sample_generator(Python Generator): The sample generator provides
                calibrate data for DataLoader, and it only returns a sample every
                time. Note that, sample_generator and batch_generator, only one
                should be set. Beisdes, sample_generator dose not support lod tensor.
            data_loader(Python Generator, Paddle.io.DataLoader, optional): The
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time.
            batch_size(int, optional): The batch size of DataLoader. Default is 10.
            batch_nums(int, optional): If batch_nums is not None, the number of
                calibrate data is batch_size*batch_nums. If batch_nums is None, use
                all data provided by sample_generator as calibrate data.
            algo(str, optional): If algo='KL', use KL-divergenc method to
                get the KL threshold for quantized activations and get the abs_max
                value for quantized weights. If algo='abs_max', get the abs max
                value for activations and weights. If algo= 'min_max', get the min
                and max value for quantized activations and weights. If algo='avg',
                get the average value among the max values for activations. If
                algo= 'hist', get the value of 'hist_percent' quantile as the threshold.
                If algo='mse', get the value which makes the quantization mse loss
                minimal. Default is KL.
            hist_percent(float, optional): The threshold of algo 'hist' for activations.
                Default is 0.99999.
            quantizable_op_type(list[str], optional): List the type of ops
                that will be quantized. Default is ["conv2d", "depthwise_conv2d",
                "mul"].
            recon_grit(str, optional): The type of reconstruction granularity.
                Currently supports ['layer-wise', 'block-wise'] types.
            is_intro_act_noise(bool, optional): Whether the noise caused by activation 
                quantization is introduced in the reconstruction process.
            learning_rate(float, optional): The learning rate of adaround method.
            is_full_quantized(bool, optional): If set is_full_quantized as True,
                apply quantization to all supported quantizable op type. If set
                is_full_quantized as False, only apply quantization to the op type
                according to the input quantizable_op_type.
            bias_correction(bool, optional): If set as True, use the bias correction
                method of https://arxiv.org/abs/1810.05723. Default is False.
            activation_bits(int): quantization bit number for activation.
            weight_bits(int, optional): quantization bit number for weights.
            activation_quantize_type(str): quantization type for activation,
                now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
                This param only specifies the fake ops in saving quantized model.
                If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
                obtained by post training quantization in fake ops. Note that, if it
                is 'abs_max', the scale will not be saved in fake ops.
            weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. This param only specifies
                the fake ops in saving quantized model, and we save the scale obtained
                by post training quantization in fake ops. Compared to 'abs_max',
                the model accuracy is usually higher when it is 'channel_wise_abs_max'.
            onnx_format(bool): Whether to export the quantized model with format of ONNX.
                Default is False.
            freeze_model(bool): Whether to convert quantized and trained ``program`` to final
                quantized ``program``. Default: True.
            skip_tensor_list(list): List of skip quant tensor name. Default: None.
            same_scale_tensor_list(list(list)): The list of tensor keep same scale in the outermost
                list, the final scale about every list is the max of the scale in the list
                of tensor. Default: None.
            optimize_model(bool, optional): If set optimize_model as True, it applies
                some passes to the model before quantization, and it supports
                `conv2d/depthwise_conv2d + bn` pass so far. Some targets require the
                weights are quantized by tensor-wise method, which means the weights
                scale for all channel are the same. However, if fuse
                `conv2d/depthwise_conv2d + bn`, the weights scale for all channel will
                be different. In address this problem, fuse the pattern before
                quantization. Default False.
            is_use_cache_file(bool, optional): This param is deprecated.
            cache_dir(str, optional): This param is deprecated.
            blocks(list[list], optional): The list of some blocks, each block is subgraph of 
                fp32 program and it will have exact 1 input operation and 1 output operation.
            block_weights_names(list[list], optional): The weight names inside every block.
        Returns:
            None

        Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization

            exe = fluid.Executor(fluid.CPUPlace())
            model_dir = path/to/fp32_model_params
            # set model_filename as None when the filename is __model__,
            # otherwise set it as the real filename
            model_filename = None
            # set params_filename as None when all parameters were saved in
            # separate files, otherwise set it as the real filename
            params_filename = None
            save_model_path = path/to/save_model_path
            # prepare the sample generator according to the model, and the
            # sample generator must return a sample every time. The reference
            # document: https://www.paddlepaddle.org.cn/documentation/docs/zh
            # /user_guides/howto/prepare_data/use_py_reader.html
            sample_generator = your_sample_generator
            batch_size = 10
            batch_nums = 10
            algo = "KL"
            quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
            rq = ReconstructionQuantization(
                        executor=exe,
                        sample_generator=sample_generator,
                        model_dir=model_dir,
                        model_filename=model_filename,
                        params_filename=params_filename,
                        batch_size=batch_size,
                        batch_nums=batch_nums,
                        algo=algo,
                        quantizable_op_type=quantizable_op_type)
            rq.quantize()
            rq.save_quantized_model(save_model_path)
        '''

        self._support_activation_quantize_type = [
            'range_abs_max', 'moving_average_abs_max', 'abs_max'
        ]
        self._support_weight_quantize_type = ['abs_max', 'channel_wise_abs_max']
        self._support_algo_type = [
            'KL', 'hist', 'avg', 'mse', 'emd', 'abs_max', 'min_max', 'ptf'
        ]
        assert recon_grit in ['layer-wise', 'block-wise']
        self._recon_grit = recon_grit
        self._is_intro_act_noise = is_intro_act_noise
        self._learning_rate = learning_rate
        self._dynamic_quantize_op_type = ['lstm']
        self._support_quantize_op_type = \
            list(set(utils._weight_supported_quantizable_op_type +
                utils._act_supported_quantizable_op_type +
                self._dynamic_quantize_op_type))

        # Check inputs
        assert executor is not None, "The executor cannot be None."
        assert any([gen is not None] for gen in [sample_generator,
            batch_generator, data_loader]), "The sample_generator, batch_generator " \
            "and data_loader cannot be None in the same time."
        if data_loader is not None:
            assert isinstance(data_loader, (io.DataLoader, type(isgeneratorfunction), reader.GeneratorLoader)), \
                "data_loader only accepts `paddle.io.DataLoader` or Generator instance."
        assert batch_size > 0, "The batch_size should be greater than 0."
        assert algo in self._support_algo_type, \
            "The algo should be KL, hist, mse, avg, abs_max, min_max or ptf."
        assert activation_quantize_type in self._support_activation_quantize_type, \
            "The activation_quantize_type ({}) should in ({}).".format(
            activation_quantize_type, self._support_activation_quantize_type)
        assert weight_quantize_type in self._support_weight_quantize_type, \
            "The weight_quantize_type ({}) shoud in ({}).".format(
            weight_quantize_type, self._support_weight_quantize_type)

        # Save input params
        self._bias_correction = bias_correction
        self._executor = executor
        self._scope = global_scope() if scope == None else scope
        self._model_dir = model_dir
        self._model_filename = model_filename
        self._params_filename = params_filename
        self._sample_generator = sample_generator
        self._batch_generator = batch_generator
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._algo = algo
        self._hist_percent = hist_percent
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._onnx_format = onnx_format
        self._clip_extra = True if self._onnx_format else False
        self._skip_tensor_list = skip_tensor_list
        self._is_full_quantize = is_full_quantize
        if is_full_quantize:
            self._quantizable_op_type = self._support_quantize_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in self._support_quantize_op_type, \
                    op_type + " is not supported for quantization."
        self._optimize_model = optimize_model

        # Define variables
        self._place = self._executor.place
        self._program = None
        self._feed_list = None
        self._fetch_list = None
        self._data_loader = data_loader

        self._out_scale_op_list = utils.QUANT_SUPPORTED_OP_TYPE_LIST
        self._quantized_weight_var_name = set()
        self._quantized_act_var_name = set()
        self._weight_op_pairs = {}
        # The vars for alog = KL or hist
        self._sampling_act_abs_min_max = {}
        self._sampling_act_histogram = {}
        self._sampling_data = {}
        self._quantized_var_threshold = {}
        self._histogram_bins = 2048
        # The vars for algo = min_max
        self._quantized_var_min = {}
        self._quantized_var_max = {}
        # The vars for algo = avg
        self._quantized_var_avg = {}
        # The best loss of algo = mse
        self._best_calibration_loss = {}
        # The threshold for algo = abs_max, mse or avg
        self._quantized_threshold = {}
        self._same_scale_tensor_list = same_scale_tensor_list
        self._freeze_model = freeze_model
        self._scale_dict = scale_dict
        self._return_graph = return_graph
        self.FLAG = False
        if self._program is not None:
            self.FLAG = True
        self._blocks = blocks
        self._block_weights_names = block_weights_names
        self._epochs = epochs
        self._scale_trainable = scale_trainable

    def quantize(self):
        '''
        Load the FP32 model, and use the calibrate data to calculate the forward-stage.
        Based on the sample data, we can get the quantization information, and obtain
        the final quantized model.

        Args:
            None
        Returns:
            the program of quantized model.
        '''
        self._load_model_data()
        self._collect_target_varnames()
        self._set_activation_persistable()

        if self._algo in ["KL", "hist"]:
            batch_id = 0
            with tqdm(
                    total=self._batch_nums,
                    bar_format=
                    'Preparation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                    ncols=80) as t:
                for data in self._data_loader():
                    self._executor.run(program=self._program,
                                       feed=data,
                                       fetch_list=self._fetch_list,
                                       return_numpy=False,
                                       scope=self._scope)
                    self._collect_activation_abs_min_max()
                    batch_id += 1
                    t.update()
                    if self._batch_nums and batch_id >= self._batch_nums:
                        break
            self._init_sampling_act_histogram()

        batch_id = 0
        with tqdm(total=self._batch_nums,
                  bar_format=
                  'Sampling stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                  ncols=80) as t:
            for data in self._data_loader():
                self._executor.run(program=self._program,
                                   feed=data,
                                   fetch_list=self._fetch_list,
                                   return_numpy=False,
                                   scope=self._scope)
                self._sampling()
                batch_id += 1
                t.update()
                if self._batch_nums and batch_id >= self._batch_nums:
                    break

        if self._algo == 'avg':
            for var_name in self._quantized_act_var_name:
                self._quantized_threshold[var_name] = \
                np.array(self._quantized_var_avg[var_name]).mean()
        if self._algo in ["KL", "hist"]:
            self._calculate_kl_hist_threshold()
        self._reset_activation_persistable()

        if self._algo in ["KL", "hist"]:
            scale_dict = self._quantized_var_threshold
        else:
            scale_dict = self._quantized_threshold

        recontruction_quanter = RecontructionQuanter(
            data_loader=self._data_loader,
            fp32_program=self._program,
            feed_list=self._feed_list,
            fetch_list=self._fetch_list,
            exe=self._executor,
            scope=self._scope,
            place=self._place,
            quantized_op_pairs=self._quantized_op_pairs,
            weight_quantize_type=self._weight_quantize_type,
            scale_dict=copy.deepcopy(scale_dict),
            blocks=self._blocks,
            block_weights_names=self._block_weights_names,
            recon_grit=self._recon_grit,
            is_intro_act_noise=self._is_intro_act_noise,
            num_iterations=self._batch_nums,
            lr=self._learning_rate,
            bias_correction=self._bias_correction,
            epochs=self._epochs,
            scale_trainable=self._scale_trainable
        )
        self._program = recontruction_quanter._run()

        if self._algo is 'min_max':
            self._save_input_threhold()
        else:
            self._update_program()

        # save out_threshold for quantized ops.
        if not self.FLAG:
            self._save_output_threshold()

        if any(op_type in self._quantizable_op_type
               for op_type in self._dynamic_quantize_op_type):
            self._collect_dynamic_quantize_op_threshold(
                self._dynamic_quantize_op_type)

        # Move sub blocks persistable var to global block
        global_block = self._program.global_block()
        for _op in global_block.ops:
            if _op.type == "while":
                _block_id = _op.attr("sub_block").id
                _block = self._program.block(_block_id)
                persistables = []
                for _name, _var in _block.vars.items():
                    if _var.persistable:
                        global_block._clone_variable(_var)
                        persistables.append(_name)
                for _name in persistables:
                    _block._remove_var(_name)
                persistables.extend(_op.input('X'))
                _op.desc.set_input("X", persistables)

        if not self._return_graph:
            return self._program
        else:
            main_graph = IrGraph(core.Graph(self._program.desc), for_test=True)
            return main_graph


    def save_quantized_model(self,
                             save_model_path,
                             model_filename=None,
                             params_filename=None):
        '''
        Save the quantized model to the disk.

        Args:
            save_model_path(str): The path to save the quantized model.
            model_filename(str, optional): If the model_filename is None,
                save the model to '__model__'. Otherwise, save the model
                to the specified filename. Default: None.
            params_filename(str, optional): If the params_filename is None,
                save params to separted files. Otherwise, save all params
                to the specified filename.
        Returns:
            None
        '''
        io.save_inference_model(dirname=save_model_path,
                                model_filename=model_filename,
                                params_filename=params_filename,
                                feeded_var_names=self._feed_list,
                                target_vars=self._fetch_list,
                                executor=self._executor,
                                main_program=self._program,
                                clip_extra=self._clip_extra)
        _logger.info("The quantized model is saved in " + save_model_path)

    def _load_model_data(self):
        '''
        Load model and set data loader.
        '''
        if self._program is None:
            _logger.info("Load model and set data loader ...")
            [self._program, self._feed_list, self._fetch_list] = \
                io.load_inference_model(dirname=self._model_dir,
                                        executor=self._executor,
                                        model_filename=self._model_filename,
                                        params_filename=self._params_filename)

        if self._optimize_model:
            self._optimize_fp32_model()

        feed_vars = [framework._get_var(str(var_name), self._program) \
            for var_name in self._feed_list]

        if self._data_loader is not None:
            self._batch_nums = self._batch_nums if self._batch_nums else len(
                self._data_loader)
            return
        self._data_loader = io.DataLoader.from_generator(feed_list=feed_vars,
                                                         capacity=3 *
                                                         self._batch_size,
                                                         iterable=True)
        if self._sample_generator is not None:
            self._data_loader.set_sample_generator(self._sample_generator,
                                                   batch_size=self._batch_size,
                                                   drop_last=True,
                                                   places=self._place)
        elif self._batch_generator is not None:
            self._data_loader.set_batch_generator(self._batch_generator,
                                                  places=self._place)
        self._batch_nums = self._batch_nums if self._batch_nums else len(
            list(self._data_loader))

    def _optimize_fp32_model(self):
        '''
        Fuse the `conv2d/depthwise_conv2d + bn` in FP32 model.
        '''
        _logger.info("Optimize FP32 model ...")
        graph = IrGraph(core.Graph(self._program.desc), for_test=True)
        graph = _remove_ctrl_vars(graph)
        graph = _apply_pass(self._scope, graph, 'conv_bn_fuse_pass')
        graph = _apply_pass(self._scope, graph, 'depthwise_conv_bn_fuse_pass')
        graph = _apply_pass(self._scope, graph, 'conv_transpose_bn_fuse_pass')
        graph = _apply_pass(self._scope, graph, 'conv_eltwiseadd_bn_fuse_pass')
        graph = _apply_pass(self._scope, graph,
                            'depthwise_conv_eltwiseadd_bn_fuse_pass')

        self._program = graph.to_program()

    def _collect_target_varnames(self):
        '''
        Collect the variable names for sampling, and set activation
        variables to be persistable.
        '''
        # TODO(juncaipeng), consider the name_scope of skip_quant
        _logger.info("Collect quantized variable names ...")
        self._quantized_op_pairs = {}

        def collect_var_name(var_name_list, persistable_var_names, op_type):
            for var_name in var_name_list:
                if var_name in persistable_var_names:
                    self._quantized_weight_var_name.add(var_name)
                    self._weight_op_pairs[var_name] = op_type
                else:
                    self._quantized_act_var_name.add(var_name)

        persistable_var_names = _all_persistable_var_names(self._program)
        for block_id in range(len(self._program.blocks)):
            for op in self._program.blocks[block_id].ops:
                # skip quant form self._skip_tensor_list
                if self._skip_tensor_list is not None:
                    for inp_name in utils._get_op_input_var_names(op):
                        if inp_name in self._skip_tensor_list:
                            op._set_attr("op_namescope", "skip_quant")

                op_type = op.type
                if self._is_full_quantize and \
                    op_type not in self._quantizable_op_type:
                    _logger.warning(op_type +
                                    " is not supported for quantization.")
                # For quantized ops, sample inputs and outputs
                if op_type in self._quantizable_op_type:
                    collect_var_name(utils._get_op_input_var_names(op),
                                     persistable_var_names, op_type)
                    collect_var_name(utils._get_op_output_var_names(op),
                                     persistable_var_names, op_type)
                    # collect quanted op output var name
                    for out_var_name in utils._get_op_output_var_names(op):
                        for in_var_name in utils._get_op_input_var_names(op):
                            if in_var_name in persistable_var_names:
                                self._quantized_op_pairs[
                                    in_var_name] = out_var_name
                # For other op, only sample output scale
                elif op_type in self._out_scale_op_list:
                    collect_var_name(utils._get_op_output_var_names(op),
                                     persistable_var_names, op_type)

    def _set_activation_persistable(self):
        '''
        Set activation variables to be persistable, so can obtain
        the tensor data in sample_data
        '''
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = True

    def _reset_activation_persistable(self):
        '''
        Reset activations to be not persistable.
        '''
        to_erase = []
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = False
                to_erase.append(var.name)

    def _sampling(self):
        '''
        Sample the min/max, abs_max or histogram in every iterations.
        '''
        if self._algo == "abs_max":
            self._sample_abs_max()
        elif self._algo == "avg":
            self._sample_avg()
        elif self._algo == "min_max":
            self._sample_min_max()
        elif self._algo == "mse":
            self._sample_mse()
        elif self._algo == "emd":
            self._sample_emd()
        elif self._algo == "ptf":
            self._sample_ptf()
        elif self._algo in ["KL", "hist"]:
            self._sample_histogram()

    def _sample_mse(self):
        if self._quantized_threshold == {}:
            for var_name in self._quantized_weight_var_name:
                var_tensor = utils.load_variable_data(self._scope, var_name)
                if self._weight_quantize_type == "abs_max":
                    abs_max_value = float(np.max(np.abs(var_tensor)))
                elif self._weight_quantize_type == "channel_wise_abs_max":
                    abs_max_value = []
                    if self._weight_op_pairs[
                            var_name] in utils._channelwise_quant_axis1_ops:
                        for i in range(var_tensor.shape[1]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[:, i]))))
                    else:
                        for i in range(var_tensor.shape[0]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[i]))))
                self._quantized_threshold[var_name] = abs_max_value
        _logger.info("MSE searching stage ...")
        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            var_tensor = var_tensor.flatten()
            abs_max_value = float(np.max(np.abs(var_tensor)))
            abs_max_value = 1e-8 if abs_max_value == 0.0 else abs_max_value
            s = 0.3
            if var_name not in self._best_calibration_loss:
                self._best_calibration_loss[var_name] = float('inf')
            while s <= 1.0:
                scale = s * abs_max_value
                s += 0.02
                bins = 2**(self._activation_bits - 1) - 1
                if self._onnx_format:
                    quant_var = np.clip(np.round(var_tensor / scale * bins),
                                        -bins - 1, bins)
                    quant_dequant_var = quant_var / bins * scale
                else:
                    quant_dequant_var = np.round(
                        np.clip(var_tensor, 0.0, scale) / scale *
                        bins) / bins * scale
                mse_loss = ((var_tensor - quant_dequant_var)**2).mean()
                if mse_loss <= self._best_calibration_loss[var_name]:
                    self._best_calibration_loss[var_name] = mse_loss
                    self._quantized_threshold[var_name] = scale

    def _sample_emd(self):
        if self._quantized_threshold == {}:
            for var_name in self._quantized_weight_var_name:
                var_tensor = utils.load_variable_data(self._scope, var_name)
                if self._weight_quantize_type == "abs_max":
                    abs_max_value = float(np.max(np.abs(var_tensor)))
                elif self._weight_quantize_type == "channel_wise_abs_max":
                    abs_max_value = []
                    if self._weight_op_pairs[
                            var_name] in utils._channelwise_quant_axis1_ops:
                        for i in range(var_tensor.shape[1]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[:, i]))))
                    else:
                        for i in range(var_tensor.shape[0]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[i]))))
                self._quantized_threshold[var_name] = abs_max_value
        _logger.info("EMD searching stage ...")
        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            var_tensor = var_tensor.flatten()
            abs_max_value = float(np.max(np.abs(var_tensor)))
            abs_max_value = 1e-8 if abs_max_value == 0.0 else abs_max_value
            s = 0.3
            if var_name not in self._best_calibration_loss:
                self._best_calibration_loss[var_name] = float('inf')
            while s <= 1.0:
                scale = s * abs_max_value
                s += 0.02
                bins = 2**(self._activation_bits - 1) - 1
                if self._onnx_format:
                    quant_var = np.clip(np.round(var_tensor / scale * bins),
                                        -bins - 1, bins)
                    quant_dequant_var = quant_var / bins * scale
                else:
                    quant_dequant_var = np.round(
                        np.clip(var_tensor, 0.0, scale) / scale *
                        bins) / bins * scale
                emd_loss = np.abs(
                    np.mean(var_tensor) - np.mean(quant_dequant_var)) + np.abs(
                        np.std(var_tensor) - np.std(quant_dequant_var))
                if emd_loss <= self._best_calibration_loss[var_name]:
                    self._best_calibration_loss[var_name] = emd_loss
                    self._quantized_threshold[var_name] = scale

    def _sample_avg(self):
        if self._quantized_threshold == {}:
            for var_name in self._quantized_weight_var_name:
                var_tensor = utils.load_variable_data(self._scope, var_name)
                if self._weight_quantize_type == "abs_max":
                    abs_max_value = float(np.max(np.abs(var_tensor)))
                elif self._weight_quantize_type == "channel_wise_abs_max":
                    abs_max_value = []
                    if self._weight_op_pairs[
                            var_name] in utils._channelwise_quant_axis1_ops:
                        for i in range(var_tensor.shape[1]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[:, i]))))
                    else:
                        for i in range(var_tensor.shape[0]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[i]))))
                self._quantized_threshold[var_name] = abs_max_value

        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            abs_max_value = float(np.max(np.abs(var_tensor)))
            if (var_name not in self._quantized_var_avg):
                self._quantized_var_avg[var_name] = []
            abs_avg_value = float(np.mean(np.max(  \
            np.abs(var_tensor.reshape(var_tensor.shape[0], -1)), axis=(1))))
            self._quantized_var_avg[var_name].append(abs_avg_value)
            continue

    def _sample_abs_max(self):
        if self._quantized_threshold == {}:
            for var_name in self._quantized_weight_var_name:
                var_tensor = utils.load_variable_data(self._scope, var_name)
                if self._weight_quantize_type == "abs_max":
                    abs_max_value = float(np.max(np.abs(var_tensor)))
                elif self._weight_quantize_type == "channel_wise_abs_max":
                    abs_max_value = []
                    if self._weight_op_pairs[
                            var_name] in utils._channelwise_quant_axis1_ops:
                        for i in range(var_tensor.shape[1]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[:, i]))))
                    else:
                        for i in range(var_tensor.shape[0]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[i]))))
                self._quantized_threshold[var_name] = abs_max_value

        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            abs_max_value = float(np.max(np.abs(var_tensor)))
            if (var_name not in self._quantized_threshold) or \
                (abs_max_value > self._quantized_threshold[var_name]):
                self._quantized_threshold[var_name] = abs_max_value

    def _sample_min_max(self):
        if self._quantized_var_min == {} and self._quantized_var_max == {}:
            for var_name in self._quantized_weight_var_name:
                var_tensor = utils.load_variable_data(self._scope, var_name)
                if self._weight_quantize_type == "abs_max":
                    min_value = float(np.min(var_tensor))
                    max_value = float(np.max(var_tensor))
                elif self._weight_quantize_type == "channel_wise_abs_max":
                    min_value = []
                    max_value = []
                    if self._weight_op_pairs[
                            var_name] in utils._channelwise_quant_axis1_ops:
                        for i in range(var_tensor.shape[1]):
                            min_value.append(float(np.min(var_tensor[:, i])))
                            max_value.append(float(np.max(var_tensor[:, i])))
                    else:
                        for i in range(var_tensor.shape[0]):
                            min_value.append(float(np.min(var_tensor[i])))
                            max_value.append(float(np.max(var_tensor[i])))
                self._quantized_var_min[var_name] = min_value
                self._quantized_var_max[var_name] = max_value

        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            min_value = float(np.min(var_tensor))
            max_value = float(np.max(var_tensor))
            if (var_name not in self._quantized_var_min) or \
                (min_value < self._quantized_var_min[var_name]):
                self._quantized_var_min[var_name] = min_value
            if (var_name not in self._quantized_var_max) or \
                (max_value > self._quantized_var_max[var_name]):
                self._quantized_var_max[var_name] = max_value

    def _sample_histogram(self):
        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            var_tensor_abs = np.abs(var_tensor)
            bins = self._sampling_act_histogram[var_name][1]
            hist, _ = np.histogram(var_tensor_abs, bins=bins)
            self._sampling_act_histogram[var_name][0] += hist

    def _sample_ptf(self):
        """
        The following code are modified from:
        https://github.com/megvii-research/FQ-ViT/
        """
        if self._quantized_threshold == {}:
            for var_name in self._quantized_weight_var_name:
                var_tensor = utils.load_variable_data(self._scope, var_name)
                if self._weight_quantize_type == "abs_max":
                    abs_max_value = float(np.max(np.abs(var_tensor)))
                elif self._weight_quantize_type == "channel_wise_abs_max":
                    abs_max_value = []
                    if self._weight_op_pairs[
                            var_name] in utils._channelwise_quant_axis1_ops:
                        for i in range(var_tensor.shape[1]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[:, i]))))
                    else:
                        for i in range(var_tensor.shape[0]):
                            abs_max_value.append(
                                float(np.max(np.abs(var_tensor[i]))))
                self._quantized_threshold[var_name] = abs_max_value

        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            abs_max_value = float(np.max(np.abs(var_tensor)))
            q_max = 2**(self._activation_bits - 1) - 1
            scale8 = abs_max_value / q_max
            scale4 = scale8 / 2
            scale2 = scale4 / 2
            scale1 = scale2 / 2
            quant_dequant_var_scale1 = np.clip(np.round(var_tensor / scale1), 0,
                                               q_max) * scale1
            quant_dequant_var_scale2 = np.clip(np.round(var_tensor / scale2), 0,
                                               q_max) * scale2
            quant_dequant_var_scale4 = np.clip(np.round(var_tensor / scale4), 0,
                                               q_max) * scale4
            quant_dequant_var_scale8 = np.clip(np.round(var_tensor / scale8), 0,
                                               q_max) * scale8
            score1 = utils.l2_loss(var_tensor, quant_dequant_var_scale1)
            score2 = utils.l2_loss(var_tensor, quant_dequant_var_scale2)
            score4 = utils.l2_loss(var_tensor, quant_dequant_var_scale4)
            score8 = utils.l2_loss(var_tensor, quant_dequant_var_scale8)
            score = [score1, score2, score4, score8]
            mask = 2**score.index(min(score))
            scale = scale1 * mask
            threshold = q_max * scale
            self._quantized_threshold[var_name] = threshold

    def _save_input_threhold(self):
        '''
        Save input threshold to the quantized op.
        '''
        assert self._algo == "min_max", \
            "The algo should be min_max to save input threshold."
        for block_id in range(len(self._program.blocks)):
            for op in self._program.blocks[block_id].ops:
                if op.type in self._quantizable_op_type:
                    for var_name in utils._get_op_input_var_names(op):
                        assert var_name in self._quantized_var_min
                        assert var_name in self._quantized_var_max
                        op._set_attr(var_name + ".min",
                                     self._quantized_var_min[var_name])
                        op._set_attr(var_name + ".max",
                                     self._quantized_var_max[var_name])
                        op._set_attr("with_quant_attr", True)

    def _collect_activation_abs_min_max(self):
        '''
        Collect the abs_min and abs_max for all activation. When algo = KL,
        get the min and max value, and then calculate the threshold.
        '''
        for var_name in self._quantized_act_var_name:
            var_tensor = utils.load_variable_data(self._scope, var_name)
            var_tensor = np.abs(var_tensor)
            min_value = float(np.min(var_tensor))
            max_value = float(np.max(var_tensor))
            if var_name not in self._sampling_act_abs_min_max:
                self._sampling_act_abs_min_max[var_name] = [
                    min_value, max_value
                ]
            else:
                if min_value < self._sampling_act_abs_min_max[var_name][0]:
                    self._sampling_act_abs_min_max[var_name][0] = min_value
                if max_value > self._sampling_act_abs_min_max[var_name][1]:
                    self._sampling_act_abs_min_max[var_name][1] = max_value

    def _init_sampling_act_histogram(self):
        '''
        Based on the min/max value, init the sampling_act_histogram.
        '''
        for var_name in self._quantized_act_var_name:
            if var_name not in self._sampling_act_histogram:
                min_val = self._sampling_act_abs_min_max[var_name][0]
                max_val = self._sampling_act_abs_min_max[var_name][1]
                hist, hist_edeges = np.histogram([],
                                                 bins=self._histogram_bins,
                                                 range=(min_val, max_val))
                self._sampling_act_histogram[var_name] = [hist, hist_edeges]

    def _calculate_kl_hist_threshold(self):
        '''
        Calculate the KL or hist threshold of quantized variables.
        '''
        _logger.info("Calculate {} threshold ...".format(self._algo))
        assert self._algo in ["KL", "hist"], "The algo should be KL or hist."

        # Abs_max threshold for weights
        for var_name in self._quantized_weight_var_name:
            weight_data = utils.load_variable_data(self._scope, var_name)
            if self._weight_quantize_type == "abs_max":
                weight_threshold = float(np.max(np.abs(weight_data)))
            elif self._weight_quantize_type == "channel_wise_abs_max":
                weight_threshold = []
                if self._weight_op_pairs[
                        var_name] in utils._channelwise_quant_axis1_ops:
                    for i in range(weight_data.shape[1]):
                        weight_threshold.append(
                            float(np.max(np.abs(weight_data[:, i]))))
                else:
                    for i in range(weight_data.shape[0]):
                        weight_threshold.append(
                            float(np.max(np.abs(weight_data[i]))))
            self._quantized_var_threshold[var_name] = weight_threshold

        for var_name in self._quantized_act_var_name:
            hist, hist_edeges = self._sampling_act_histogram[var_name]
            if self._algo == "KL":
                bin_width = hist_edeges[1] - hist_edeges[0]
                self._quantized_var_threshold[var_name] = \
                    cal_kl_threshold(hist, bin_width, self._activation_bits)
            elif self._algo == "hist":
                self._quantized_var_threshold[var_name] = \
                    self._get_hist_scaling_factor(hist, hist_edeges)

    def _update_program(self):
        '''
        Use QuantizationTransformPass and AddQuantDequantPass to insert
        fake_quantize, fake_dequantize and fake_quant_dequant op.
        Besides, save all threshold to the scale var node.
        '''
        _logger.info("Update the program ...")
        graph = IrGraph(core.Graph(self._program.desc), for_test=True)

        # use QuantizationTransformPass to insert fake_quant/fake_dequantize op
        major_quantizable_op_types = []
        for op_type in utils._weight_supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                major_quantizable_op_types.append(op_type)
        if not self._onnx_format:
            transform_pass = QuantizationTransformPass(
                scope=self._scope,
                place=self._place,
                weight_bits=self._weight_bits,
                activation_bits=self._activation_bits,
                activation_quantize_type=self._activation_quantize_type,
                weight_quantize_type=self._weight_quantize_type,
                quantizable_op_type=major_quantizable_op_types)
        else:
            transform_pass = QuantizationTransformPassV2(
                scope=self._scope,
                place=self._place,
                weight_bits=self._weight_bits,
                activation_bits=self._activation_bits,
                activation_quantize_type=self._activation_quantize_type,
                weight_quantize_type=self._weight_quantize_type,
                quantizable_op_type=major_quantizable_op_types)

        for sub_graph in graph.all_sub_graphs():
            # Insert fake_quant/fake_dequantize op must in test graph, so
            # set per graph's _for_test is True.
            sub_graph._for_test = True
            transform_pass.apply(sub_graph)

        # use AddQuantDequantPass to insert fake_quant_dequant op
        minor_quantizable_op_types = []
        for op_type in utils._act_supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                minor_quantizable_op_types.append(op_type)
        if not self._onnx_format:
            add_quant_dequant_pass = AddQuantDequantPass(
                scope=self._scope,
                place=self._place,
                quantizable_op_type=minor_quantizable_op_types)
        else:
            add_quant_dequant_pass = AddQuantDequantPassV2(
                scope=self._scope,
                place=self._place,
                quantizable_op_type=minor_quantizable_op_types,
                is_full_quantized=True)

        for sub_graph in graph.all_sub_graphs():
            sub_graph._for_test = True
            add_quant_dequant_pass.apply(sub_graph)

        # save threshold to scale var node
        if self._scale_dict is None:
            if self._algo in ["KL", "hist"]:
                scale_dict = self._quantized_var_threshold
            else:
                scale_dict = self._quantized_threshold

            if self._same_scale_tensor_list is not None:
                for tensor_list in self._same_scale_tensor_list:
                    max_scale = None
                    tmp_tensor_list = []
                    for tensor_name in tensor_list:
                        if '#' in tensor_name:
                            real_tensor_name, opera, scalar = tensor_name.split(
                                '#')
                            if real_tensor_name not in scale_dict.keys():
                                continue
                            if opera == '*':
                                scale_dict[real_tensor_name] = float(
                                    scale_dict[real_tensor_name]) * float(
                                        scalar)
                            elif opera == '/':
                                scale_dict[real_tensor_name] = float(
                                    scale_dict[real_tensor_name]) / float(
                                        scalar)
                            max_scale = scale_dict[
                                real_tensor_name] if max_scale is None else max(
                                    max_scale, scale_dict[real_tensor_name])
                        else:
                            if tensor_name not in scale_dict.keys():
                                continue
                            max_scale = scale_dict[
                                tensor_name] if max_scale is None else max(
                                    max_scale, scale_dict[tensor_name])

                    for tensor_name in tensor_list:
                        if '#' in tensor_name:
                            real_tensor_name, opera, scalar = tensor_name.split(
                                '#')
                            if real_tensor_name not in scale_dict.keys():
                                continue
                            if opera == '*':
                                scale_dict[
                                    real_tensor_name] = max_scale / float(
                                        scalar)
                            elif opera == '/':
                                scale_dict[
                                    real_tensor_name] = max_scale * float(
                                        scalar)
                        else:
                            if tensor_name not in scale_dict.keys():
                                continue
                            scale_dict[tensor_name] = max_scale
            self._scale_dict = scale_dict

        for key, val in self._scale_dict.items():
            utils.set_variable_data(self._scope, self._place, key + "@scale",
                                    np.array([val], dtype=np.float32))
            utils.set_variable_data(self._scope, self._place,
                                    key + ".quant_dequant@scale",
                                    np.array([val], dtype=np.float32))

        if not self._onnx_format:
            # apply QuantizationFreezePass, and obtain the final quant model
            if self._freeze_model:
                freeze_pass = QuantizationFreezePass(
                    scope=self._scope,
                    place=self._place,
                    bias_correction=self._bias_correction,
                    weight_bits=self._weight_bits,
                    round_type='adaround',
                    activation_bits=self._activation_bits,
                    weight_quantize_type=self._weight_quantize_type,
                    quantizable_op_type=major_quantizable_op_types)

                for sub_graph in graph.all_sub_graphs():
                    sub_graph._for_test = True
                    freeze_pass.apply(sub_graph)
        else:
            quant_weight_pass = QuantWeightPass(self._scope, self._place)
            for sub_graph in graph.all_sub_graphs():
                sub_graph._for_test = True
                quant_weight_pass.apply(sub_graph)

        self._program = graph.to_program()

    def _save_output_threshold(self):
        '''
        Save output threshold to the quantized op.
        '''
        self._calibration_scales = {}

        def save_info(op_node, out_var_name, threshold_map, out_info_name,
                      quantized_type):
            assert out_var_name in threshold_map, \
                "The output ({}) of {} node does not have threshold.".format(
                out_var_name, op_node.type)
            if self._onnx_format:
                # For easy extension, every var_node set a dict to save parameters of quant.
                self._calibration_scales[var_name] = {}
                self._calibration_scales[var_name]['scale'] = threshold_map[
                    var_name]
            else:
                op_node._set_attr(out_info_name, threshold_map[var_name])
                op_node._set_attr("with_quant_attr", True)
                if op_node.type in self._quantizable_op_type:
                    op._set_attr("quantization_type", quantized_type)

        def analysis_and_save_info(op_node, out_var_name):
            argname_index = utils._get_output_name_index(op_node, out_var_name)
            assert argname_index is not None, \
                out_var_name + " is not the output of the op"
            if self._algo == "KL":
                # For compatibility, we save output threshold by two methods.
                save_info(op_node, out_var_name, self._quantized_var_threshold,
                          "out_threshold", "post_kl")
                save_info(
                    op_node, out_var_name, self._quantized_var_threshold,
                    argname_index[0] + str(argname_index[1]) + "_threshold",
                    "post_kl")
            elif self._algo == "hist":
                # For compatibility, we save output threshold by two methods.
                save_info(op_node, out_var_name, self._quantized_var_threshold,
                          "out_threshold", "post_hist")
                save_info(
                    op_node, out_var_name, self._quantized_var_threshold,
                    argname_index[0] + str(argname_index[1]) + "_threshold",
                    "post_hist")

            elif self._algo in ["avg", "abs_max", "mse", "emd", "ptf"]:
                save_info(op_node, out_var_name, self._quantized_threshold,
                          "out_threshold", "post_" + str(self._algo))
                save_info(
                    op_node, out_var_name, self._quantized_threshold,
                    argname_index[0] + str(argname_index[1]) + "_threshold",
                    "post_" + str(self._algo))
            elif self._algo == "min_max":
                save_info(op_node, out_var_name, self._quantized_var_min,
                          "out_min", "post_min_max")
                save_info(op_node, out_var_name, self._quantized_var_max,
                          "out_max", "post_min_max")

        for block_id in range(len(self._program.blocks)):
            for op in self._program.blocks[block_id].ops:
                if op.type in (self._quantizable_op_type +
                               self._out_scale_op_list):
                    out_var_names = utils._get_op_output_var_names(op)
                    for var_name in out_var_names:
                        analysis_and_save_info(op, var_name)

    def _collect_dynamic_quantize_op_threshold(self, target_ops_type):
        """
        Collect and save the weight threshold for dynamic quantize ops,
        such as lstm and gru.
        Args:
            target_ops_type(list): the op type of target ops
        Returns:
            None
        """

        target_ops = []
        for index in range(self._program.num_blocks):
            for op in self._program.block(index).ops:
                if op.type in target_ops_type:
                    target_ops.append(op)

        quantization_type = str("post_" + self._algo).lower()
        persistable_var_names = _all_persistable_var_names(self._program)
        for op in target_ops:
            for var_name in utils._get_op_input_var_names(op):
                if var_name in persistable_var_names:
                    var_data = utils.load_variable_data(self._scope, var_name)
                    threshold = float(np.max(np.abs(var_data)))
                    argname, index = utils._get_input_name_index(op, var_name)
                    op._set_attr(argname + str(index) + "_threshold", threshold)
                    op._set_attr("quantization_type", quantization_type)
                    op._set_attr("bit_length", self._weight_bits)
                    op._set_attr("with_quant_attr", True)

    def _get_hist_scaling_factor(self, hist, hist_edges):
        '''
        Using the hist method to get the scaling factor.
        '''
        threshold_rate = self._hist_percent
        hist = hist / float(sum(hist))
        hist_sum = 0
        hist_index = 0
        for i in range(len(hist)):
            hist_sum += hist[i]
            if hist_sum >= threshold_rate:
                hist_index = i + 1
                break
        bin_width = hist_edges[1] - hist_edges[0]
        return (hist_index - 0.5) * bin_width



class RecontructionQuanter(object):

    def __init__(self,
                 data_loader,
                 fp32_program,
                 feed_list,
                 fetch_list,
                 exe,
                 scope,
                 place,
                 quantized_op_pairs,
                 weight_quantize_type,
                 scale_dict,
                 blocks,
                 block_weights_names,
                 recon_grit,
                 is_intro_act_noise,
                 num_iterations=1000,
                 lr=0.1,
                 bias_correction=False,
                 epochs=20,
                 scale_trainable=False
                 ):
        '''
        Rounding Optimizer, used to optimize the rounding policy
        by reconstructing the intermediate output.

        Args:
            data_loader(Python Generator, Paddle.io.DataLoader, optional): The
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time.
            executor(fluid.Executor): The executor to load, run and save the
                quantized model.
            scope(fluid.Scope, optional): The scope of the program, use it to load 
                and save variables. If scope=None, get scope by global_scope(). 
            place(CPUPlace()|CUDAPlace(N)): This parameter represents
                                                    paddle run on which device.
            quantized_op_pairs(dict, optional): Mapping of op's weight name 
                and output var name, where key of dict is the weight name of 
                op, and value is the output var name of op.
            weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. This param only specifies
                the fake ops in saving quantized model, and we save the scale obtained
                by post training quantization in fake ops. Compared to 'abs_max',
                the model accuracy is usually higher when it is 'channel_wise_abs_max'.
            scale_dict(dict, optional): Mapping of var's name and var's scales, where key 
                of dict is the var name, and value is the quant scales of var.
            recon_grit(str, optional): The type of reconstruction granularity.
                Currently supports ['layer-wise', 'block-wise'] types.
            is_intro_act_noise(bool, optional): Whether the noise caused by activation 
                quantization is introduced in the reconstruction process.
            blocks(list[list], optional): The list of some blocks, each block is subgraph of 
                fp32 program and it will have exact 1 input operation and 1 output operation.
            block_weights_names(list[list], optional): The weight names inside every block.
            lr(float, optional): The learning rate of Rounding Optimizer.
            bias_correction(bool, optional): If set as True, use the bias correction
                method of https://arxiv.org/abs/1810.05723. Default is False.

        Returns:
            None
        '''

        assert recon_grit in ['layer-wise', 'block-wise']
        if recon_grit=='block-wise':
            assert blocks is not None, "The blocks cannot be None."
            assert block_weights_names is not None, "The block_weights_names cannot be None."
        self._is_intro_act_noise = is_intro_act_noise
        self._program = fp32_program
        self._data_loader = data_loader
        self._recon_grit = recon_grit
        self._feed_list = feed_list
        self._fetch_list = fetch_list
        self._exe = exe
        self._scope = scope
        self._place = place
        self._quantized_op_pairs = quantized_op_pairs
        self._weight_var_names = list(self._quantized_op_pairs.keys())
        self._weight_quantize_type = weight_quantize_type
        self._scale_dict = scale_dict
        self._num_iterations = num_iterations
        self._epochs = epochs
        self._lr = lr
        self._blocks = blocks
        self._block_weights_names = block_weights_names
        self._bias_correction = bias_correction
        if self._recon_grit == 'layer-wise':
            blocks, block_weights_names = self._get_layers()
            self._blocks = blocks
            self._block_weights_names = block_weights_names
        self._scale_trainable = scale_trainable

    def _get_layers(self):
        blocks = []
        block_weights_names = []
        persistable_var_names = self._all_persistable_var_names()
        self._input_weight_pairs = {}
        for block_id in range(len(self._program.blocks)):
            for op in self._program.blocks[block_id].ops:
                in_var_names = utils._get_op_input_var_names(op)
                for in_var_name in in_var_names:
                    if in_var_name in persistable_var_names:
                        in_var_names.remove(in_var_name)
                        self._input_weight_pairs[in_var_name] = in_var_names
                        break
        for name in self._weight_var_names:
            block_weights_names.append([name])
            block_ = []
            block_.append(self._input_weight_pairs[name][0])
            block_.append(self._quantized_op_pairs[name])
            blocks.append(block_)
        return blocks, block_weights_names
    
    def _preprocess(self): 
        data_name_map = {}
        for name in self._feed_list:
            data_name_map[name] = name
        self._student_program = self._program.clone()
        merge(
            self._program,
            self._student_program,
            data_name_map,
            self._place,
            teacher_scope=None,
            name_prefix="teacher_",
            merge_feed=True)
        for name in self._weight_var_names:
            weight_np = utils.load_variable_data(self._scope, name)
            scale = self._scale_dict[name]
            weight_np_floor = np.floor(utils.quant_tensor(weight_np, scale))
            utils.set_variable_data(self._scope, self._place, name, weight_np_floor)
        self._graph = GraphWrapper(self._student_program)
        
        if self._is_intro_act_noise:
            self._insert_drop_quant_dequant()
        self._insert_soft_rounding()
        self._isolate_blocks()
    
    def _run(self):
        self._preprocess()
        startup_program = paddle.static.Program()
        for k in range(len(self._blocks)):
            block_ = self._blocks[k]
            names = self._block_weights_names[k]
            tmp_program = self._student_program.clone()
            quant_op_out_name = block_[1]
            with paddle.static.program_guard(tmp_program, startup_program):
                loss_function = RecontructionQuanterLoss(tmp_program, names)
                quant_op_out_name = block_[1]
                student_var = tmp_program.global_block().var(quant_op_out_name)
                teacher_var = tmp_program.global_block().var("teacher_"+quant_op_out_name)
                scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=20, eta_min=2, T_max=2000, verbose=True)
                total_loss, recon_loss, round_loss = loss_function.get_loss(student_var, teacher_var, scheduler)
                train_fetches_loss = {"total_loss":total_loss, "recon_loss":recon_loss, "round_loss":round_loss}
                optimizer = paddle.optimizer.Adam(learning_rate=self._lr)
                optimizer.minimize(total_loss)

            self._exe.run(startup_program)
            start_time = time.time()
            prev_start_time = start_time
            for epoch in range(self._epochs):
                for i, data in enumerate(self._data_loader()):
                    prev_start_time = start_time
                    start_time = time.time()
                    out = self._exe.run(
                        tmp_program,
                        feed=data,
                        fetch_list=[v.name for v in train_fetches_loss.values()],
                        return_numpy=True)
                    _logger.info(
                        "Iter {:d}, lr {}, total_loss {:.5f}, recon_loss {:.5f}, round_loss {:.5f}, time {:.5f}s"
                        .format(epoch, self._lr, np.mean(out[0]), np.mean(out[1]), np.mean(out[2]), start_time - prev_start_time))
                    sys.stdout.flush()
                    if i == self._num_iterations:
                        break
        self._update_weights_to_int()
        if self._bias_correction:
            self._bias_correction_w()
        return self._program

    def _init_alpha(self, name, scale):
        _tensor = utils.load_variable_data(self._scope, "teacher_"+name)
        tensor_scaled = utils.quant_tensor(_tensor, scale)
        tensor_floor = np.floor(tensor_scaled)
        tensor = tensor_scaled - tensor_floor
        alpha = -np.log((ZETA - GAMMA) / (tensor - GAMMA) - 1)
        return alpha

    def _soft_rounding(self, weight, scale, weight_bits=8):
        """
        Define network of soft rounding.
        Args:
        weight: The quanted weight with dtype=float32
        """
        bnt = (1 << (weight_bits - 1)) - 1
        def _dequant(x, scale):
            s = (scale+1e-8)/bnt
            dequant_x = s * x 
            return dequant_x
        quantized_weight = paddle.static.data(shape=weight.shape,
                                            dtype=weight.dtype,
                                            name=weight.name+'_quant')

        v = paddle.static.create_parameter(shape=weight.shape,
                                            dtype=weight.dtype,
                                            name=weight.name+".alpha",
                                            default_initializer=fluid.initializer.NumpyArrayInitializer(self._alpha))

        h_v = paddle.clip(paddle.nn.functional.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0, 1)

        if self._weight_quantize_type=='channel_wise_abs_max':
            scale_var = paddle.static.create_parameter(
                    dtype=weight.dtype,
                    shape=weight.shape,
                    name=weight.name+'.scale',
                    default_initializer=fluid.initializer.NumpyArrayInitializer(scale),
                )
        else:
            scale_var = scale
        w = _dequant(quantized_weight+h_v, scale_var)
        return w

    def _insert_soft_rounding(self):
        for name in self._weight_var_names:
            weight = self._graph.var(name)
            scale = self._scale_dict[name]
            shape = weight.shape()
            self._alpha = self._init_alpha(name, scale)
            if self._weight_quantize_type=='channel_wise_abs_max':
                scale = np.array(scale)
                scale = scale.reshape(scale.shape[0], 1)
                if len(shape)==2:
                    scale = scale.repeat(shape[0], axis=0)
                else:
                    scale = scale.repeat(shape[1]*shape[2]*shape[3], axis=1)
                scale = scale.reshape(shape)
            self._insert_func(var=weight, scale=scale, func="_soft_rounding")

    def _drop_quant_dequant(self, inputs, scale, weight_bits=8):
        x = paddle.static.data(shape=inputs.shape,
                                dtype=inputs.dtype,
                                name=inputs.name+'.tmp')
        bnt = (1 << (weight_bits - 1)) - 1
        scale = scale / bnt
        dequantized_tensor = paddle.round(x / scale) * scale
        quant_noise = x - dequantized_tensor
        random_noise = paddle.nn.functional.dropout(quant_noise, p=0.5)
        return x + random_noise

    def _insert_drop_quant_dequant(self):
        for op in self._graph.ops():
            if op.type() in ['conv2d', 'depthwise_conv2d', 'mul']:
                if op.type() in ['conv2d', 'depthwise_conv2d']:
                    if op.inputs("Filter")[0].name().startswith("teacher"):
                        break
                    else:
                        input = op.inputs("Input")[0]
                if op.type() in ['mul']:
                    if op.inputs("Y")[0].name().startswith("teacher"):
                        break
                    else:
                        input = op.inputs("X")[0]
                if input.name() in self._scale_dict.keys():
                    self._insert_func(var=input, scale=self._scale_dict[input.name()], func="_drop_quant_dequant")

    def _insert_func(self, var, scale, func):
        program = var._graph.program
        ops = var.outputs()
        inputs = var._var
        startup_program = paddle.static.Program()
        new_program = paddle.static.Program()
        with paddle.static.program_guard(new_program, startup_program):
            if func=="_soft_rounding":
                out = self._soft_rounding(inputs, scale)
            elif func=="_drop_quant_dequant":
                out = self._drop_quant_dequant(inputs, scale)
        self._exe.run(startup_program)
        #create var in program
        for new_var in new_program.list_vars():
            if new_var.name == var._var.name+'_quant' or  new_var.name == var._var.name+'.tmp':
                continue
            elif new_var.name == var._var.name+'.alpha':
                program.global_block().create_parameter(
                name=new_var.name,
                shape=new_var.shape,
                dtype=new_var.dtype,
                type=new_var.type,
                stop_gradient=new_var.stop_gradient)
            elif new_var.name == var._var.name+'.scale':
                program.global_block().create_parameter(
                name=new_var.name,
                shape=new_var.shape,
                dtype=new_var.dtype,
                type=new_var.type,
                stop_gradient=True,
                trainable=self._scale_trainable)
            else:
                if func=="_soft_rounding":
                    program.global_block().create_var(
                    name=new_var.name+'.rounding',
                    shape=new_var.shape,
                    dtype=new_var.dtype,
                    type=new_var.type,
                    persistable=new_var.persistable,
                    stop_gradient=new_var.stop_gradient)
                else:
                    program.global_block().create_var(
                    name=new_var.name,
                    shape=new_var.shape,
                    dtype=new_var.dtype,
                    type=new_var.type,
                    persistable=new_var.persistable,
                    stop_gradient=new_var.stop_gradient)
        op_list = new_program.global_block().ops
        op_list = list(reversed(op_list))
        block = var._var.block
        #prepend new_program's op in program
        for _op in ops:
            if _op.type() not in ['conv2d', 'depthwise_conv2d', 'mul']:
                continue
            idx = block.ops.index(_op._op)
            for op in op_list:
                # _attrs = op.all_attrs()
                _type = op.type
                _attrs={
                    'use_mkldnn': False,
                    'with_quant_attr' :False}
                if _type=='clip':
                    _attrs={
                        'use_mkldnn': False,
                        'with_quant_attr' :False,
                        'max':op.attr('max'),
                        'min':op.attr('min')}
                elif _type=='scale':
                    _attrs={
                        'use_mkldnn': False,
                        'with_quant_attr' :False,
                        'scale': op.attr('scale'),
                        'bias_after_scale':op.attr('bias_after_scale')}
                elif _type=='elementwise_mul':
                    _attrs={
                        'use_mkldnn': False,
                        'with_quant_attr' :False,
                        'Scale_out':op.attr('Scale_out'),
                        'Scale_x':op.attr('Scale_x'),
                        'Scale_y':op.attr('Scale_y'),
                        'axis':op.attr('axis')}
                
                if func=="_soft_rounding":
                    _outputs = {'Out':op.output('Out')[0]+'.rounding'}
                    if _type=="elementwise_add":
                        _inputs = {
                            'X': var._var,     #replace tmp var conv.weight_quant with var conv.weight
                            'Y': op.input('Y')[0]+'.rounding',
                            }
                    elif _type=="elementwise_mul":
                        _inputs = {
                            'X':op.input('X')[0]+'.rounding',
                            'Y':op.input('Y')[0]+'.rounding',
                            }
                    elif (_type=='scale' and op.input('X')[0].endswith('scale')) or _type=='sigmoid':
                        _inputs = {'X':op.input('X')[0]}
                    else:
                        _inputs = {'X':op.input('X')[0]+'.rounding'}
                elif func=="_drop_quant_dequant":
                    if _type=='dropout':
                        _outputs = {'Out':op.output('Out')[0],
                                    'Mask':op.output('Mask')[0]}
                    else:
                        _outputs = {'Out':op.output('Out')[0]}

                    if _type=='elementwise_add' or _type=='elementwise_sub':
                        _inputs = {
                            'X': var._var,     #replace tmp var conv.weight_quant with var conv.weight
                            'Y': op.input('Y'),
                            }
                    elif _type=='scale' and op.input('X')[0]==inputs.name+'.tmp':
                        _inputs = {'X': var._var}
                    else:
                        _inputs = {'X':op.input('X')[0]}

                block._insert_op(
                    idx,
                    type=_type,
                    attrs=_attrs,
                    inputs=_inputs,
                    outputs=_outputs,
                )
        for op in ops:
            if op.type() not in ['conv2d', 'depthwise_conv2d', 'mul']:
                continue
            if op.type() in ['conv2d', 'depthwise_conv2d'] and op.inputs('Filter')[0].name().startswith('teacher'):
                continue
            if op.type() in ['mul'] and op.inputs('Y')[0].name().startswith('teacher'):
                continue        
            if func=='_soft_rounding':
                op._op._rename_input(inputs.name, out.name+'.rounding')
            else:
                op._op._rename_input(inputs.name, out.name)
                
    def _isolate_blocks(self):
        starts = [block[0] for block in self._blocks]
        var2duplications = self._duplicate_vars(starts)
        for vars_ in var2duplications.values():
            for var_ in vars_:
                var_.stop_gradients = True

    def _duplicate_vars(self, var_names):
        result = {}
        for var_name in var_names:
            var = self._graph.var(var_name)
            result[var_name] = self._duplicate_var(var)
        return result

    def _duplicate_var(self, var):
        vars = []
        block = var._var.block
        index = 0
        for op in var.outputs():
            var_ = var._var
            op_ = op._op
            duplicated_var = block.create_var(name=var_.name+".assign"+str(index),
                                        type=var_.type,
                                        shape=var_.shape,
                                        dtype=var_.dtype)
            vars.append(duplicated_var)
            index += 1
            idx = block.ops.index(op_)
            block._insert_op(idx,
                            type="assign",
                            inputs={"X": var_},
                            outputs={"Out": duplicated_var})
            op_._rename_input(var_.name, duplicated_var.name)
        return vars

    def _update_weights_to_int(self):
        for weight_var_name in self._weight_var_names:
            alpha_tensor = utils.load_variable_data(self._scope, weight_var_name+'.alpha')
            h_alpha_tensor = self._compute_soft_rounding_np(alpha_tensor)
            weight_quant_tensor = utils.load_variable_data(self._scope, weight_var_name)
            utils.set_variable_data(self._scope, self._place, weight_var_name, np.round(weight_quant_tensor+h_alpha_tensor))

            x = utils.load_variable_data(self._scope, weight_var_name)
            print(weight_var_name, x.flatten()[:20])

    def _bias_correction_w(self):
        for weight_var_name in self._weight_var_names:
            weight_var_tensor = utils.load_variable_data(self._scope, "teacher_"+weight_var_name)
            weight_quant_tensor = utils.load_variable_data(self._scope, weight_var_name)
            scale = self._scale_dict[weight_var_name]
            final_weight_tensor = utils.bias_correction_w(
                weight_var_tensor,
                weight_quant_tensor,
                scale,
                quant_axis=0,
                weight_bits=8)   
            utils.set_variable_data(self._scope, self._place, weight_var_name, final_weight_tensor)

    def _compute_soft_rounding_np(self, alpha_v):
        return np.clip(utils.stable_sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA,
                    a_min=0,
                    a_max=1)

            
    def _all_persistable_var_names(self):
        persistable_var_names = []
        for var in self._program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)
        return persistable_var_names



class RecontructionQuanterLoss(object):
    
    def __init__(self,
                 program,
                 weight_block_names=None,
                 round_loss_mode='relaxation',
                 rec_loss_mode='mse',
                 beta_mode='const',
                 weight=0.1,):
        """
        The loss function of Rounding Optimizer.

        Args:
            program(Program): The student program.
            weight_block_names(list, optional): The weight names inside a block.
            round_loss_mode(str): The rounding loss function mode.
            rec_loss_mode(str): The reconstruction loss function mode.
            beta_mode(str): The parameter beta mode.
        Returns:
            total_loss(Variable): The sum of rounding loss and reconstruction loss.
            rec_loss(Variable): The reconstruction loss.
            round_loss(Variable): The rounding loss.
        """
        self.program = program
        self.round_loss_mode = round_loss_mode
        self.weight = weight
        self.rec_loss_mode = rec_loss_mode
        self.weight_block_names = weight_block_names
        self.beta_mode = beta_mode

    def compute_soft_rounding(self, alpha_v):
        return paddle.clip(paddle.nn.functional.sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA, 0, 1)

    def get_loss(self, student_tensor, teacher_tensor, scheduler):
        if self.rec_loss_mode == 'mse':
            rec_loss = paddle.nn.functional.mse_loss(student_tensor, teacher_tensor)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        if self.beta_mode == 'const':
            self.beta = 3
        else:
            self.beta = scheduler.get_lr()

        if self.round_loss_mode == 'relaxation':
            round_loss = 0.0
            for name in self.weight_block_names:
                alpha_v = self.program.global_block().var(name+'.alpha')
                h_v = self.compute_soft_rounding(alpha_v)
                round_loss += self.weight * paddle.sum(-paddle.pow(paddle.abs(2 * h_v-1), self.beta) + 1)
        else:
            raise NotImplementedError
        total_loss = rec_loss+round_loss
        return total_loss, rec_loss, round_loss

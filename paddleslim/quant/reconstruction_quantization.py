#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import logging
import math
import os
import re
import shutil
import sys
import time

import numpy as np
import paddle

import paddle.fluid as fluid
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle.fluid.contrib.slim.quantization import utils

from ..dist import merge
from ..core.graph_wrapper import GraphWrapper
from ..common import get_logger

__all__ = ['ReconstructionQuantization', ]

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s', )

GAMMA = -0.1
ZETA = 1.1


class Collections(object):
    def __init__(self, **kwargs):
        self._config = dict()
        for k, v in kwargs.items():
            self._config[k] = v

    def _get_config(self):
        return self._config


class ReconstructionQuantization(PostTrainingQuantization):
    """
    Utilizing reconstruction quantization method to quantize the FP32 model,
    and it uses calibrate data to get the quantization information for all
    quantized variables.
    """

    def __init__(self, PTQCollections, RSQCollections):
        '''
        Args:
            PTQCollections(Collections): The parameters set required for post training quantization.
            RSQCollections(Collections): The parameters set required for reconstruction quantization.    
        Returns:
            None
        '''
        super().__init__(**PTQCollections._get_config())
        self._config = RSQCollections._get_config()

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
            self._preparation()
        self._sampling_threshold()
        self._calculate_threshold()
        self._reset_activation_persistable()
        self._reconstruction()
        self._postprocessing()
        if not self._return_graph:
            return self._program
        else:
            main_graph = IrGraph(core.Graph(self._program.desc), for_test=True)
            return main_graph

    def _preparation(self):
        batch_id = 0
        with utils.tqdm(
                total=self._batch_nums,
                bar_format='Preparation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                ncols=80, ) as t:
            for data in self._data_loader():
                self._executor.run(
                    program=self._program,
                    feed=data,
                    fetch_list=self._fetch_list,
                    return_numpy=False,
                    scope=self._scope, )
                self._collect_activation_abs_min_max()
                batch_id += 1
                t.update()
                if self._batch_nums and batch_id >= self._batch_nums:
                    break
        self._init_sampling_act_histogram()

    def _sampling_threshold(self):
        batch_id = 0
        with utils.tqdm(
                total=self._batch_nums,
                bar_format='Sampling stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                ncols=80, ) as t:
            for data in self._data_loader():
                self._executor.run(
                    program=self._program,
                    feed=data,
                    fetch_list=self._fetch_list,
                    return_numpy=False,
                    scope=self._scope, )
                self._sampling()
                batch_id += 1
                t.update()
                if self._batch_nums and batch_id >= self._batch_nums:
                    break

    def _calculate_threshold(self):
        if self._algo == 'avg':
            for var_name in self._quantized_act_var_name:
                self._quantized_threshold[var_name] = \
                    np.array(self._quantized_var_avg[var_name]).mean()
            self._scale_dict = self._quantized_threshold

        elif self._algo in ["KL", "hist"]:
            self._calculate_kl_hist_threshold()
            self._scale_dict = self._quantized_var_threshold
        else:
            self._scale_dict = self._quantized_threshold

    def _reconstruction(self):
        reconstruction_quanter = ReconstructionQuanter(
            data_loader=self._data_loader,
            fp32_program=self._program,
            feed_list=self._feed_list,
            fetch_list=self._fetch_list,
            exe=self._executor,
            scope=self._scope,
            place=self._place,
            quantized_op_pairs=self._quantized_op_pairs,
            weight_quantize_type=self._weight_quantize_type,
            scale_dict=copy.deepcopy(self._scale_dict),
            regions=self._config['regions'],
            region_weights_names=self._config['region_weights_names'],
            recon_level=self._config['recon_level'],
            simulate_activation_quant=self._config['simulate_activation_quant'],
            num_iterations=self._batch_nums,
            lr=self._learning_rate,
            bias_correction=self._bias_correction,
            epochs=self._config['epochs'],
            scale_trainable=self._config['scale_trainable'])
        self._program = reconstruction_quanter._run()

    def _postprocessing(self):
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
                self._dynamic_quantize_op_type, )

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


class ReconstructionQuanter(object):
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
                 regions,
                 region_weights_names,
                 recon_level,
                 simulate_activation_quant,
                 num_iterations=1000,
                 lr=0.1,
                 bias_correction=False,
                 epochs=20,
                 scale_trainable=False,
                 drop_prob=0.5):
        '''
        Reconstruction Quanter, used to optimize the rounding policy
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
            recon_level(str, optional): The type of reconstruction granularity.
                Currently support ['layer-wise', 'region-wise'] types. Default is layer-wise.
            simulate_activation_quant(bool, optional): Whether we need the noise caused by activation 
                quantization during the reconstruction process.
            regions(list[list], optional): The list of some regions, each region is a subgraph of
                fp32 program and it will have exact 1 input operation and 1 output operation. When 
                the recon-level is region, the reconstruction loss of each region is minimized.
                Default is None.
            region_weights_names(list[list], optional): The weight names inside every region.
                Default is None.
            lr(float, optional): The learning rate of Reconstruction Quanter. Default is 0.1.
            bias_correction(bool, optional): If set as True, use the bias correction
                method of https://arxiv.org/abs/1810.05723. Default is False.
            scale_trainable: Wether weight‘s scale is trainable. Default is False.
            drop_prob: The dropout probability of activation quantization, and it is valid only if 
                simulate_activation_quant is True. Default is 0.5.
        Returns:
            None
        '''

        assert recon_level in [
            'layer-wise', 'region-wise'
        ], "recon_level must be one of the ['layer-wise', 'region-wise'],but received: {}".format(
            recon_level)
        if recon_level == 'region-wise':
            assert regions is not None, "The regions cannot be None."
            assert region_weights_names is not None, "The region_weights_names cannot be None."
        self._simulate_activation_quant = simulate_activation_quant
        self._program = fp32_program
        self._data_loader = data_loader
        self._recon_level = recon_level
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
        self._regions = regions
        self._region_weights_names = region_weights_names
        self._bias_correction = bias_correction
        if self._recon_level == 'layer-wise':
            regions, region_weights_names = self._get_layers()
            self._regions = regions
            self._region_weights_names = region_weights_names
        self._scale_trainable = scale_trainable
        self._drop_prob = drop_prob

    def _get_layers(self):
        regions = []
        region_weights_names = []
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
            region_weights_names.append([name])
            region_ = []
            region_.append(self._input_weight_pairs[name][0])
            region_.append(self._quantized_op_pairs[name])
            regions.append(region_)
        return regions, region_weights_names

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
            merge_feed=True, )
        for name in self._weight_var_names:
            weight_np = utils.load_variable_data(self._scope, name)
            scale = self._scale_dict[name]
            weight_np_floor = np.floor(utils.quant_tensor(weight_np, scale))
            utils.set_variable_data(
                self._scope,
                self._place,
                name,
                weight_np_floor, )
        self._graph = GraphWrapper(self._student_program)

        if self._simulate_activation_quant:
            self._insert_drop_quant_dequant()
        self._insert_soft_rounding()
        self._isolate_regions()

    def _run(self):
        self._preprocess()
        startup_program = paddle.static.Program()
        for k in range(len(self._regions)):
            region_ = self._regions[k]
            names = self._region_weights_names[k]
            tmp_program = self._student_program.clone()
            quant_op_out_name = region_[1]
            with paddle.static.program_guard(tmp_program, startup_program):
                loss_function = ReconstructionQuanterLoss(tmp_program, names)
                quant_op_out_name = region_[1]
                student_var = tmp_program.global_block().var(quant_op_out_name)
                teacher_var = tmp_program.global_block().var("teacher_" +
                                                             quant_op_out_name)
                scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                    learning_rate=20,
                    eta_min=2,
                    T_max=2000,
                    verbose=True, )
                total_loss, recon_loss, round_loss = loss_function.get_loss(
                    student_var,
                    teacher_var,
                    scheduler, )
                train_fetches_loss = {
                    "total_loss": total_loss,
                    "recon_loss": recon_loss,
                    "round_loss": round_loss,
                }
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
                        fetch_list=[
                            v.name for v in train_fetches_loss.values()
                        ],
                        return_numpy=True, )
                    _logger.info(
                        "Iter {:d}, lr {}, total_loss {:.5f}, recon_loss {:.5f}, round_loss {:.5f}, time {:.5f}s"
                        .format(epoch, self._lr,
                                np.mean(out[0]),
                                np.mean(out[1]),
                                np.mean(out[2]),
                                start_time - prev_start_time), )
                    sys.stdout.flush()
                    if i == self._num_iterations:
                        break
        self._update_weights_to_int()
        if self._bias_correction:
            self._bias_correction_w()
        return self._program

    def _init_alpha(self, name, scale):
        _tensor = utils.load_variable_data(self._scope, "teacher_" + name)
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
            s = (scale + 1e-8) / bnt
            dequant_x = s * x
            return dequant_x

        quantized_weight = paddle.static.data(
            shape=weight.shape,
            dtype=weight.dtype,
            name=weight.name + '_quant', )

        v = paddle.static.create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            name=weight.name + ".alpha",
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                self._alpha, ), )

        h_v = paddle.clip(
            paddle.nn.functional.sigmoid(v) * (ZETA - GAMMA) + GAMMA,
            0,
            1, )

        if self._weight_quantize_type == 'channel_wise_abs_max':
            scale_var = paddle.static.create_parameter(
                dtype=weight.dtype,
                shape=weight.shape,
                name=weight.name + '.scale',
                default_initializer=fluid.initializer.NumpyArrayInitializer(
                    scale, ), )
        else:
            scale_var = scale
        w = _dequant(quantized_weight + h_v, scale_var)
        return w

    def _insert_soft_rounding(self):
        for name in self._weight_var_names:
            weight = self._graph.var(name)
            scale = self._scale_dict[name]
            shape = weight.shape()
            self._alpha = self._init_alpha(name, scale)
            if self._weight_quantize_type == 'channel_wise_abs_max':
                scale = np.array(scale)
                scale = scale.reshape(scale.shape[0], 1)
                if len(shape) == 2:
                    scale = scale.repeat(shape[0], axis=0)
                else:
                    scale = scale.repeat(shape[1] * shape[2] * shape[3], axis=1)
                scale = scale.reshape(shape)
            self._insert_func(var=weight, scale=scale, func="_soft_rounding")

    def _drop_quant_dequant(self, inputs, scale, weight_bits=8):
        x = paddle.static.data(
            shape=inputs.shape,
            dtype=inputs.dtype,
            name=inputs.name + '.tmp', )
        bnt = (1 << (weight_bits - 1)) - 1
        scale = scale / bnt
        dequantized_tensor = paddle.round(x / scale) * scale
        quant_noise = x - dequantized_tensor
        random_noise = paddle.nn.functional.dropout(
            quant_noise, p=self._drop_prob)
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
                    self._insert_func(
                        var=input,
                        scale=self._scale_dict[input.name()],
                        func="_drop_quant_dequant", )

    def _insert_func(self, var, scale, func):
        program = var._graph.program
        ops = var.outputs()
        inputs = var._var
        startup_program = paddle.static.Program()
        new_program = paddle.static.Program()
        with paddle.static.program_guard(new_program, startup_program):
            if func == "_soft_rounding":
                out = self._soft_rounding(inputs, scale)
            elif func == "_drop_quant_dequant":
                out = self._drop_quant_dequant(inputs, scale)
        self._exe.run(startup_program)
        # create var in program
        for new_var in new_program.list_vars():
            if new_var.name == var._var.name + '_quant' or new_var.name == var._var.name + '.tmp':
                continue
            elif new_var.name == var._var.name + '.alpha':
                program.global_block().create_parameter(
                    name=new_var.name,
                    shape=new_var.shape,
                    dtype=new_var.dtype,
                    type=new_var.type,
                    stop_gradient=new_var.stop_gradient, )
            elif new_var.name == var._var.name + '.scale':
                program.global_block().create_parameter(
                    name=new_var.name,
                    shape=new_var.shape,
                    dtype=new_var.dtype,
                    type=new_var.type,
                    stop_gradient=True,
                    trainable=self._scale_trainable, )
            else:
                if func == "_soft_rounding":
                    program.global_block().create_var(
                        name=new_var.name + '.rounding',
                        shape=new_var.shape,
                        dtype=new_var.dtype,
                        type=new_var.type,
                        persistable=new_var.persistable,
                        stop_gradient=new_var.stop_gradient, )
                else:
                    program.global_block().create_var(
                        name=new_var.name,
                        shape=new_var.shape,
                        dtype=new_var.dtype,
                        type=new_var.type,
                        persistable=new_var.persistable,
                        stop_gradient=new_var.stop_gradient, )
        op_list = new_program.global_block().ops
        op_list = list(reversed(op_list))
        block = var._var.block
        # prepend new_program's op in program
        for _op in ops:
            if _op.type() not in ['conv2d', 'depthwise_conv2d', 'mul']:
                continue
            idx = block.ops.index(_op._op)
            for op in op_list:
                # _attrs = op.all_attrs()
                _type = op.type
                _attrs = {
                    'use_mkldnn': False,
                    'with_quant_attr': False,
                }
                if _type == 'clip':
                    _attrs = {
                        'use_mkldnn': False,
                        'with_quant_attr': False,
                        'max': op.attr('max'),
                        'min': op.attr('min'),
                    }
                elif _type == 'scale':
                    _attrs = {
                        'use_mkldnn': False,
                        'with_quant_attr': False,
                        'scale': op.attr('scale'),
                        'bias_after_scale': op.attr('bias_after_scale'),
                    }
                elif _type == 'elementwise_mul':
                    _attrs = {
                        'use_mkldnn': False,
                        'with_quant_attr': False,
                        'Scale_out': op.attr('Scale_out'),
                        'Scale_x': op.attr('Scale_x'),
                        'Scale_y': op.attr('Scale_y'),
                        'axis': op.attr('axis'),
                    }

                if func == "_soft_rounding":
                    _outputs = {'Out': op.output('Out')[0] + '.rounding'}
                    if _type == "elementwise_add":
                        _inputs = {
                            'X': var.
                            _var,  # replace tmp var conv.weight_quant with var conv.weight
                            'Y': op.input('Y')[0] + '.rounding',
                        }
                    elif _type == "elementwise_mul":
                        _inputs = {
                            'X': op.input('X')[0] + '.rounding',
                            'Y': op.input('Y')[0] + '.rounding',
                        }
                    elif (_type == 'scale' and
                          op.input('X')[0].endswith('scale')
                          ) or _type == 'sigmoid':
                        _inputs = {'X': op.input('X')[0]}
                    else:
                        _inputs = {'X': op.input('X')[0] + '.rounding'}
                elif func == "_drop_quant_dequant":
                    if _type == 'dropout':
                        _outputs = {
                            'Out': op.output('Out')[0],
                            'Mask': op.output('Mask')[0],
                        }
                    else:
                        _outputs = {'Out': op.output('Out')[0]}

                    if _type == 'elementwise_add' or _type == 'elementwise_sub':
                        _inputs = {
                            'X': var.
                            _var,  # replace tmp var conv.weight_quant with var conv.weight
                            'Y': op.input('Y'),
                        }
                    elif _type == 'scale' and op.input('X')[
                            0] == inputs.name + '.tmp':
                        _inputs = {'X': var._var}
                    else:
                        _inputs = {'X': op.input('X')[0]}

                block._insert_op(
                    idx,
                    type=_type,
                    attrs=_attrs,
                    inputs=_inputs,
                    outputs=_outputs, )
        for op in ops:
            if op.type() not in ['conv2d', 'depthwise_conv2d', 'mul']:
                continue
            if op.type() in ['conv2d', 'depthwise_conv2d'] and op.inputs(
                    'Filter')[0].name().startswith('teacher'):
                continue
            if op.type() in ['mul'] and op.inputs('Y')[0].name().startswith(
                    'teacher'):
                continue
            if func == '_soft_rounding':
                op._op._rename_input(inputs.name, out.name + '.rounding')
            else:
                op._op._rename_input(inputs.name, out.name)

    def _isolate_regions(self):
        starts = [region[0] for region in self._regions]
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
            duplicated_var = block.create_var(
                name=var_.name + ".assign" + str(index),
                type=var_.type,
                shape=var_.shape,
                dtype=var_.dtype, )
            vars.append(duplicated_var)
            index += 1
            idx = block.ops.index(op_)
            block._insert_op(
                idx,
                type="assign",
                inputs={"X": var_},
                outputs={"Out": duplicated_var}, )
            op_._rename_input(var_.name, duplicated_var.name)
        return vars

    def _update_weights_to_int(self):
        for weight_var_name in self._weight_var_names:
            alpha_tensor = utils.load_variable_data(
                self._scope,
                weight_var_name + '.alpha', )
            h_alpha_tensor = self._compute_soft_rounding_np(alpha_tensor)
            weight_quant_tensor = utils.load_variable_data(
                self._scope,
                weight_var_name, )
            utils.set_variable_data(
                self._scope,
                self._place,
                weight_var_name,
                np.round(weight_quant_tensor + h_alpha_tensor, ), )

    def _bias_correction_w(self):
        for weight_var_name in self._weight_var_names:
            weight_var_tensor = utils.load_variable_data(
                self._scope,
                "teacher_" + weight_var_name, )
            weight_quant_tensor = utils.load_variable_data(
                self._scope,
                weight_var_name, )
            scale = self._scale_dict[weight_var_name]
            final_weight_tensor = utils.bias_correction_w(
                weight_var_tensor,
                weight_quant_tensor,
                scale,
                quant_axis=0,
                weight_bits=8, )
            utils.set_variable_data(
                self._scope,
                self._place,
                weight_var_name,
                final_weight_tensor, )

    def _compute_soft_rounding_np(self, alpha_v):
        return np.clip(
            utils.stable_sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA,
            a_min=0,
            a_max=1, )

    def _all_persistable_var_names(self):
        persistable_var_names = []
        for var in self._program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)
        return persistable_var_names


class ReconstructionQuanterLoss(object):
    def __init__(self,
                 program,
                 weight_region_names=None,
                 round_loss_type='relaxation',
                 rec_loss_type='mse',
                 beta_type='const',
                 weight=0.1):
        """
        The loss function of Rounding Optimizer.

        Args:
            program(Program): The student program.
            weight_region_names(list, optional): The weight names inside a region.
            round_loss_type(str): The type of rounding loss function.
            rec_loss_type(str): The type of reconstruction loss function.
            beta_type(str): The type of hyper-parameter beta.
        Returns:
            total_loss(Variable): The sum of rounding loss and reconstruction loss.
            rec_loss(Variable): The reconstruction loss.
            round_loss(Variable): The rounding loss.
        """
        self.program = program
        self.round_loss_type = round_loss_type
        self.weight = weight
        self.rec_loss_type = rec_loss_type
        self.weight_region_names = weight_region_names
        self.beta_type = beta_type

    def compute_soft_rounding(self, alpha_v):
        return paddle.clip(
            paddle.nn.functional.sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA, 0,
            1)

    def get_loss(self, student_tensor, teacher_tensor, scheduler):
        if self.rec_loss_type == 'mse':
            rec_loss = paddle.nn.functional.mse_loss(
                student_tensor,
                teacher_tensor, )
        else:
            raise ValueError(
                'Not supported reconstruction loss function: {}'.format(
                    self.rec_loss, ), )

        if self.beta_type == 'const':
            self.beta = 3
        else:
            self.beta = scheduler.get_lr()

        if self.round_loss_type == 'relaxation':
            round_loss = 0.0
            for name in self.weight_region_names:
                alpha_v = self.program.global_block().var(name + '.alpha')
                h_v = self.compute_soft_rounding(alpha_v)
                round_loss += self.weight * \
                    paddle.sum(-paddle.pow(paddle.abs(2 * h_v-1), self.beta) + 1)
        else:
            raise NotImplementedError
        total_loss = rec_loss + round_loss
        return total_loss, rec_loss, round_loss


def quant_recon_static(executor,
                       model_dir,
                       quantize_model_path,
                       batch_generator=None,
                       sample_generator=None,
                       data_loader=None,
                       model_filename=None,
                       params_filename=None,
                       save_model_filename='model.pdmodel',
                       save_params_filename='model.pdiparams',
                       batch_size=1,
                       batch_nums=None,
                       scope=None,
                       algo='hist',
                       recon_level='layer-wise',
                       simulate_activation_quant=False,
                       hist_percent=0.9999,
                       bias_correction=False,
                       quantizable_op_type=[
                           "conv2d",
                           "depthwise_conv2d",
                           "mul",
                           "matmul",
                           "matmul_v2",
                       ],
                       is_full_quantize=False,
                       weight_bits=8,
                       activation_bits=8,
                       activation_quantize_type='range_abs_max',
                       weight_quantize_type='channel_wise_abs_max',
                       optimize_model=False,
                       onnx_format=False,
                       skip_tensor_list=None,
                       is_use_cache_file=False,
                       cache_dir="./temp_recon_quantization",
                       regions=None,
                       region_weights_names=None,
                       epochs=20,
                       scale_trainable=False,
                       drop_prob=0.5):
    """
    The function utilizes static post training quantization method to
    quantize the fp32 model. It uses calibrate data to calculate the
    scale factor of quantized variables, and inserts fake quantization
    and dequantization operators to obtain the quantized model.

    Args:
        executor(paddle.static.Executor): The executor to load, run and save the
            quantized model.
        model_dir(str): The path of fp32 model that will be quantized, and
            the model and params that saved by ``paddle.static.io.save_inference_model``
            are under the path.
        quantize_model_path(str): The path to save quantized model using api
            ``paddle.static.io.save_inference_model``.
        batch_generator(Python Generator): The batch generator provides
            calibrate data for DataLoader, and it returns a batch every
            time. For sample_generator and batch_generator, only one
            can be set. Beisdes, batch_generator supports lod tensor.
        sample_generator(Python Generator): The sample generator provides
            calibrate data for DataLoader, and it only returns a sample every time.
        data_loader(Python Generator, Paddle.io.DataLoader, optional): The
            Generator or Dataloader provides calibrate data, and it could
            return a batch every time.
        model_filename(str, optional): The name of model file. If parameters
            are saved in separate files, set it as 'None'. Default: 'None'.
        params_filename(str, optional): The name of params file.
            When all parameters are saved in a single file, set it
            as filename. If parameters are saved in separate files,
            set it as 'None'. Default : 'None'.
        save_model_filename(str): The name of model file to save the quantized inference program.  Default: 'model.pdmodel'.
        save_params_filename(str): The name of file to save all related parameters.
            If it is set None, parameters will be saved in separate files. Default: 'model.pdiparams'.
        batch_size(int, optional): The batch size of DataLoader, default is 1.
        batch_nums(int, optional): If batch_nums is not None, the number of calibrate
            data is 'batch_size*batch_nums'. If batch_nums is None, use all data
            generated by sample_generator  as calibrate data.
        scope(paddle.static.Scope, optional): The scope to run program, use it to load
            and save variables. If scope is None, will use paddle.static.global_scope().
        algo(str, optional): If algo='KL', use KL-divergenc method to
            get the scale factor. If algo='hist', use the hist_percent of histogram
            to get the scale factor. If algo='mse', search for the best scale factor which
            makes the mse loss minimal. Use one batch of data for mse is enough. If
            algo='avg', use the average of abs_max values  to get the scale factor. If
            algo='abs_max', use abs_max method to get the scale factor. Default: 'hist'.
        recon_level(str, optional): The type of reconstruction granularity.
            Currently support ['layer-wise', 'region-wise'] types. Default is layer-wise.
        simulate_activation_quant(bool, optional): Whether we need the noise caused by activation 
            quantization during the reconstruction process. Default is False.
        hist_percent(float, optional): The percentile of histogram for algo hist.Default:0.9999.
        bias_correction(bool, optional): Bias correction method of https://arxiv.org/abs/1810.05723.
            Default: False.
        quantizable_op_type(list[str], optional): The list of op types
            that will be quantized. Default: ["conv2d", "depthwise_conv2d", "mul"].
        weight_bits(int, optional): quantization bit number for weights.
        activation_bits(int): quantization bit number for activation.
            activation_quantize_type(str): quantization type for activation,
            now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
            This parameter only specifies the fake ops in quantized model.
            If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
            obtained by post training quantization in fake ops. If it
            is 'abs_max', the scale will not be saved in fake ops.
        weight_quantize_type(str): quantization type for weights,
            support 'abs_max' and 'channel_wise_abs_max'. Compared to 'abs_max',
            the model accuracy is usually higher when using 'channel_wise_abs_max'.
        is_full_quantize(bool): if True, apply quantization to all supported quantizable op type.
            If False, only apply quantization to the input quantizable_op_type. Default is False.
        optimize_model(bool, optional): If set optimize_model as True, it applies some
            passes to optimize the model before quantization. So far, the place of
            executor must be cpu it supports fusing batch_norm into convs.
        onnx_format(bool): Whether to export the quantized model with format of ONNX. Default is False.
        skip_tensor_list(list): List of skip quant tensor name.
        is_use_cache_file(bool): This param is deprecated.
        cache_dir(str): This param is deprecated.
        epochs: The number of steps in the reconstruction proces. Default is 20.
        scale_trainable: Wether weight‘s scale is trainable. Default is False.
        drop_prob: The dropout probability of activation quantization, and it is valid only if 
            simulate_activation_quant is True. Default is 0.5.
        regions(list[list], optional): The list of some regions, each region is a subgraph of
            fp32 program and it will have exact 1 input operation and 1 output operation. When 
            the recon-level is region, the reconstruction loss of each region is minimized.
            Default is None.
        region_weights_names(list[list], optional): The weight names inside every region.
            Default is None.
    Returns:
        None
    """

    PTQCollections = Collections(
        executor=executor,
        sample_generator=sample_generator,
        batch_generator=batch_generator,
        data_loader=data_loader,
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        batch_size=batch_size,
        batch_nums=batch_nums,
        scope=scope,
        algo=algo,
        hist_percent=hist_percent,
        bias_correction=bias_correction,
        quantizable_op_type=quantizable_op_type,
        is_full_quantize=is_full_quantize,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        activation_quantize_type=activation_quantize_type,
        weight_quantize_type=weight_quantize_type,
        onnx_format=onnx_format,
        skip_tensor_list=skip_tensor_list,
        optimize_model=optimize_model,
        round_type='adaround')

    RSQCollections = Collections(
        recon_level=recon_level,
        simulate_activation_quant=simulate_activation_quant,
        regions=regions,
        region_weights_names=region_weights_names,
        epochs=epochs,
        scale_trainable=scale_trainable)

    reconstruction_quantization = ReconstructionQuantization(
        PTQCollections=PTQCollections, RSQCollections=RSQCollections)

    reconstruction_quantization.quantize()
    reconstruction_quantization.save_quantized_model(
        quantize_model_path,
        model_filename=save_model_filename,
        params_filename=save_params_filename)

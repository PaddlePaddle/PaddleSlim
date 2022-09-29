import numpy as np
import time
import sys
import logging
import paddle
import paddle.fluid as fluid
import six
import math
import copy
from ..dist import merge
from ..core.graph_wrapper import GraphWrapper
from ..common import get_logger
from paddle.fluid.contrib.slim.quantization import utils

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')
GAMMA = -0.1
ZETA = 1.1
__all__ = ['RoundingOptimizer', ]


class RoundingOptimizerLoss(object):
    def __init__(self,
                 program,
                 weight_block_names=None,
                 round_loss='relaxation',
                 weight=0.1,
                 rec_loss='mse',
                 beta_mode='const'):

        self.program = program
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.weight_block_names = weight_block_names
        self.beta_mode = beta_mode

    def compute_soft_rounding(self, alpha_v):
        return paddle.clip(
            paddle.nn.functional.sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA, 0,
            1)

    def get_loss(self, student_tensor, teacher_tensor, scheduler):
        if self.rec_loss == 'mse':
            rec_loss = paddle.nn.functional.mse_loss(student_tensor,
                                                     teacher_tensor)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.
                             format(self.rec_loss))

        if self.beta_mode == 'const':
            self.beta = 3
        else:
            self.beta = scheduler.get_lr()

        if self.round_loss == 'relaxation':
            round_loss = 0.0
            for name in self.weight_block_names:
                alpha_v = self.program.global_block().var(name + '.alpha')
                h_v = self.compute_soft_rounding(alpha_v)
                round_loss += self.weight * paddle.sum(-paddle.pow(
                    paddle.abs(2 * h_v - 1), self.beta) + 1)
        else:
            raise NotImplementedError
        total_loss = rec_loss + round_loss
        return total_loss, rec_loss, round_loss


class RoundingOptimizer(object):
    def __init__(
            self,
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
            round_type,
            num_iterations=1000,
            lr=0.1,
            bias_correction=False,
            epochs=20, ):

        assert round_type in ['adaround', 'brecq', 'qdrop']
        if round_type in ['brecq', 'qdrop']:
            assert blocks is not None, "The blocks cannot be None."
            assert block_weights_names is not None, "The block_weights_names cannot be None."
        self._program = fp32_program
        self._data_loader = data_loader
        self._round_type = round_type
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
        if round_type in ['adaround']:
            blocks, block_weights_names = self._get_layers()
            self._blocks = blocks
            self._block_weights_names = block_weights_names

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
            utils.set_variable_data(self._scope, self._place, name,
                                    weight_np_floor)
        self._graph = GraphWrapper(self._student_program)

        if self._round_type == 'qdrop':
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
                loss_function = RoundingOptimizerLoss(tmp_program, names)
                quant_op_out_name = block_[1]
                student_var = tmp_program.global_block().var(quant_op_out_name)
                teacher_var = tmp_program.global_block().var("teacher_" +
                                                             quant_op_out_name)
                scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                    learning_rate=20, eta_min=2, T_max=2000, verbose=True)
                total_loss, recon_loss, round_loss = loss_function.get_loss(
                    student_var, teacher_var, scheduler)
                train_fetches_loss = {
                    "total_loss": total_loss,
                    "recon_loss": recon_loss,
                    "round_loss": round_loss
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
                    out = self._exe.run(tmp_program,
                                        feed=data,
                                        fetch_list=[
                                            v.name
                                            for v in train_fetches_loss.values()
                                        ],
                                        return_numpy=True)
                    _logger.info(
                        "Iter {:d}, lr {}, total_loss {:.5f}, recon_loss {:.5f}, round_loss {:.5f}, time {:.5f}s"
                        .format(epoch, self._lr,
                                np.mean(out[0]),
                                np.mean(out[1]),
                                np.mean(out[2]), start_time - prev_start_time))
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
            shape=weight.shape, dtype=weight.dtype, name=weight.name + '_quant')

        v = paddle.static.create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            name=weight.name + ".alpha",
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                self._alpha))

        h_v = paddle.clip(
            paddle.nn.functional.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0, 1)

        if self._weight_quantize_type == 'channel_wise_abs_max':
            scale_var = paddle.static.create_parameter(
                dtype=weight.dtype,
                shape=weight.shape,
                name=weight.name + '.scale',
                default_initializer=fluid.initializer.NumpyArrayInitializer(
                    scale), )
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
            shape=inputs.shape, dtype=inputs.dtype, name=inputs.name + '.tmp')
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
                    self._insert_func(
                        var=input,
                        scale=self._scale_dict[input.name()],
                        func="_drop_quant_dequant")

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
        #create var in program
        for new_var in new_program.list_vars():
            if new_var.name == var._var.name + '_quant' or new_var.name == var._var.name + '.tmp':
                continue
            elif new_var.name == var._var.name + '.alpha':
                program.global_block().create_parameter(
                    name=new_var.name,
                    shape=new_var.shape,
                    dtype=new_var.dtype,
                    type=new_var.type,
                    stop_gradient=new_var.stop_gradient)
            elif new_var.name == var._var.name + '.scale':
                program.global_block().create_parameter(
                    name=new_var.name,
                    shape=new_var.shape,
                    dtype=new_var.dtype,
                    type=new_var.type,
                    stop_gradient=True,
                    trainable=False)
            else:
                if func == "_soft_rounding":
                    program.global_block().create_var(
                        name=new_var.name + '.rounding',
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
                _attrs = {'use_mkldnn': False, 'with_quant_attr': False}
                if _type == 'clip':
                    _attrs = {
                        'use_mkldnn': False,
                        'with_quant_attr': False,
                        'max': op.attr('max'),
                        'min': op.attr('min')
                    }
                elif _type == 'scale':
                    _attrs = {
                        'use_mkldnn': False,
                        'with_quant_attr': False,
                        'scale': op.attr('scale'),
                        'bias_after_scale': op.attr('bias_after_scale')
                    }
                elif _type == 'elementwise_mul':
                    _attrs = {
                        'use_mkldnn': False,
                        'with_quant_attr': False,
                        'Scale_out': op.attr('Scale_out'),
                        'Scale_x': op.attr('Scale_x'),
                        'Scale_y': op.attr('Scale_y'),
                        'axis': op.attr('axis')
                    }

                if func == "_soft_rounding":
                    _outputs = {'Out': op.output('Out')[0] + '.rounding'}
                    if _type == "elementwise_add":
                        _inputs = {
                            'X': var.
                            _var,  #replace tmp var conv.weight_quant with var conv.weight
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
                            'Mask': op.output('Mask')[0]
                        }
                    else:
                        _outputs = {'Out': op.output('Out')[0]}

                    if _type == 'elementwise_add' or _type == 'elementwise_sub':
                        _inputs = {
                            'X': var._var,
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
            duplicated_var = block.create_var(
                name=var_.name + ".assign" + str(index),
                type=var_.type,
                shape=var_.shape,
                dtype=var_.dtype)
            vars.append(duplicated_var)
            index += 1
            idx = block.ops.index(op_)
            block._insert_op(
                idx,
                type="assign",
                inputs={"X": var_},
                outputs={"Out": duplicated_var})
            op_._rename_input(var_.name, duplicated_var.name)
        return vars

    def _update_weights_to_int(self):
        for weight_var_name in self._weight_var_names:
            alpha_tensor = utils.load_variable_data(self._scope,
                                                    weight_var_name + '.alpha')
            h_alpha_tensor = self._compute_soft_rounding_np(alpha_tensor)
            weight_quant_tensor = utils.load_variable_data(self._scope,
                                                           weight_var_name)
            utils.set_variable_data(
                self._scope, self._place, weight_var_name,
                np.round(weight_quant_tensor + h_alpha_tensor))

    def _bias_correction_w(self):
        for weight_var_name in self._weight_var_names:
            weight_var_tensor = utils.load_variable_data(
                self._scope, "teacher_" + weight_var_name)
            weight_quant_tensor = utils.load_variable_data(self._scope,
                                                           weight_var_name)
            scale = self._scale_dict[weight_var_name]
            final_weight_tensor = utils.bias_correction_w(
                weight_var_tensor,
                weight_quant_tensor,
                scale,
                quant_axis=0,
                weight_bits=8)
            utils.set_variable_data(self._scope, self._place, weight_var_name,
                                    final_weight_tensor)

    def _compute_soft_rounding_np(self, alpha_v):
        return np.clip(
            utils.stable_sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA,
            a_min=0,
            a_max=1)

    def _all_persistable_var_names(self):
        persistable_var_names = []
        for var in self._program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)
        return persistable_var_names

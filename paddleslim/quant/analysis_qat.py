# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import pickle
import copy
import logging
import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from ..common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["AnalysisQAT"]


class AnalysisQAT(object):
    def __init__(self,
                 quant_model_dir,
                 float_model_dir,
                 model_filename=None,
                 params_filename=None,
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 qat_metric=None,
                 eval_function=None,
                 data_loader=None,
                 save_dir='analysis_results',
                 resume=False):
        '''
        AnalysisQAT provides to analysis the sensitivity of each op in the model.
        
        Args:
            quant_model_dir(str): the path of INT8 model that quantized through QAT
            float_model_dir(str): the path of FP32 model that is the base model of quant_model
            model_filename(str, optional): the model file name of the model
            params_filename(str, optional): the parameter file name of the model
            quantizable_op_type(list of str, optional): the type of op that will be analyzed
            eval_function(function): eval function, define by yourself to return the metric of the inference program, can be used to judge the metric of quantized model. 
            data_loader(Python Generator, Paddle.io.DataLoader, optional): the
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time
            save_dir(str, optional): the output dir that stores the analyzed information
            resume(bool, optional): When break off while ananlyzing, could resume analysis program and load already analyzed information.
        '''
        if model_filename is None:
            model_filename = 'model.pdmodel'
        if params_filename is None:
            params_filename = 'model.pdiparams'
        self.quant_model_dir = quant_model_dir
        self.float_model_dir = float_model_dir
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.quantizable_op_type = quantizable_op_type
        self.qat_metric = qat_metric
        self.eval_function = eval_function
        self.save_dir = save_dir
        self.checkpoint_name = os.path.join(save_dir, 'analysis_checkpoint.pkl')
        self.nonquant_layer_metrics = {}
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        devices = paddle.device.get_device().split(':')[0]
        self.places = paddle.device._convert_to_place(devices)
        executor = paddle.static.Executor(self.places)
        [program, self.feed_list,
         self.fetch_list] = paddle.fluid.io.load_inference_model(
             self.quant_model_dir, executor, self.model_filename,
             self.params_filename)
        _logger.info('Loaded model from: {}'.format(quant_model_dir))

        graph = IrGraph(core.Graph(program.desc), for_test=True)

        # find all inputs for each quantizable op
        self.tobe_removed_input_lists = []
        sorted_ops = graph.topology_sort()
        for op_node in sorted_ops:
            op_name = op_node.name()
            if op_name in quantizable_op_type:
                inputs = op_node.op().input_arg_names()
                for i in inputs:
                    if 'quantized' in i:
                        self.tobe_removed_input_lists.append(inputs)
                        break

        #self.data_loader = wrap_dataloader(data_loader, self.feed_list)
        if self.qat_metric is None:
            _logger.info('Calculating the metric of QAT model...')
            self.qat_metric = self.eval_function(
                executor, program, self.feed_list, self.fetch_list) * 100
            _logger.info('The metric of QAT model is {}'.format(
                round(self.qat_metric, 4)))
        executor.close()

    def save_checkpoint(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.checkpoint_name, 'wb') as f:
            pickle.dump(self.nonquant_layer_metrics, f)
        _logger.info('Save checkpoint to {}.'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            _logger.info('Checkpoint path {} does not exist.'.format(
                self.checkpoint_name))
            return False
        with open(self.checkpoint_name, 'rb') as f:
            self.nonquant_layer_metrics = pickle.load(f)
        _logger.info('Load checkpoint from {}.'.format(self.checkpoint_name))
        return True

    def metric_error_analyse(self):
        executor = paddle.static.Executor(self.places)

        float_scope = paddle.static.Scope()
        quant_scope = paddle.static.Scope()

        for idx, input_list in enumerate(self.tobe_removed_input_lists):
            w_idx = 0 if 'w_0' in input_list[0] else 1
            layer_name = input_list[w_idx].split('.quantized.dequantized')[0]
            _logger.info(
                'Checking {}/{} quant model: without quant layer {}'.format(
                    idx + 1, len(self.tobe_removed_input_lists), layer_name))

            input_rename_map = {}
            output_rename_map = {}
            with paddle.static.scope_guard(float_scope):
                [float_program, feed_target_names,
                 fetch_targets] = paddle.fluid.io.load_inference_model(
                     self.float_model_dir, executor, self.model_filename,
                     self.params_filename)

            with paddle.static.scope_guard(quant_scope):
                [program, self.feed_list,
                 self.fetch_list] = paddle.fluid.io.load_inference_model(
                     self.quant_model_dir, executor, self.model_filename,
                     self.params_filename)
            program_copy = program.clone()
            graph = IrGraph(core.Graph(program_copy.desc), for_test=True)
            removed = []
            for op_node in graph.all_op_nodes():
                if op_node.id() in removed:
                    continue
                in_names = op_node.input_arg_names()
                out_names = op_node.output_arg_names()
                if len(out_names) == 1 and out_names[0] in input_list:
                    in_var = graph._find_node_by_name(op_node.inputs,
                                                      op_node.input('X')[0])
                    out_var = graph._find_node_by_name(op_node.outputs,
                                                       op_node.output('Y')[0])
                    if 'quantized' in in_var.name():
                        # act
                        for op in graph.all_op_nodes():
                            o_ns = op.output_arg_names()
                            if len(o_ns) == 1 and o_ns[0] == in_var.name():
                                in_var_1 = graph._find_node_by_name(
                                    op.inputs, op.input('X')[0])
                                graph.safe_remove_nodes(op)
                                removed.append(op.id())
                                input_rename_map[out_var.node] = in_var_1
                    else:
                        # weight
                        with paddle.static.scope_guard(float_scope):
                            float_weight = np.array(
                                float_scope.find_var(in_var.name()).get_tensor(
                                ))
                        with paddle.static.scope_guard(quant_scope):
                            quant_scope.find_var(in_var.name()).get_tensor(
                            ).set(float_weight, paddle.fluid.CPUPlace())
                        input_rename_map[out_var.node] = in_var
                    graph.safe_remove_nodes(op_node)
                    removed.append(op_node.id())
                    output_rename_map[in_var.node] = out_var

            for op_node in graph.all_op_nodes():
                if op_node.id() in removed:
                    continue
                for var in op_node.inputs:
                    if var.node in input_rename_map:
                        old_in = var
                        new_in = input_rename_map[var.node]
                        graph.update_input_link(old_in, new_in, op_node)
                        _logger.info(
                            f'relink {op_node.name()} \'s input node from {old_in.name()} to {new_in.name()}.'
                        )
                for var in op_node.outputs:
                    if var.node in output_rename_map:
                        old_out = var
                        new_out = output_rename_map[var.node]
                        graph.update_input_link(old_out, new_out, op_node)
                        _logger.info(
                            f'relink {op_node.name()} \'s output node from {old_out.name()} to {new_out.name()}.'
                        )

            saved_program = graph.to_program()
            with paddle.static.scope_guard(quant_scope):
                _logger.info('Skip quant {}, evaluating....'.format(layer_name))
                metric = self.eval_function(executor, saved_program,
                                            self.feed_list,
                                            self.fetch_list) * 100
                self.nonquant_layer_metrics[layer_name] = metric
                _logger.info(
                    'When skip quant {}, the metric is {}, the diff is {}'.
                    format(layer_name,
                           round(metric, 4), round(metric - self.qat_metric,
                                                   4)))
            self.save_checkpoint()

        executor.close()
        self.sensitivity_ranklist = sorted(
            self.nonquant_layer_metrics,
            key=self.nonquant_layer_metrics.get,
            reverse=True)
        _logger.info('Finished computing the sensitivity of the model.')
        for name in self.sensitivity_ranklist:
            _logger.info("without quant layer name: {}, eval metric: {}".format(
                name, self.nonquant_layer_metrics[name]))

        analysis_file = os.path.join(self.save_dir, "analysis.txt")
        with open(analysis_file, "w") as analysis_ret_f:
            for name in self.sensitivity_ranklist:
                analysis_ret_f.write(
                    "without layer name: {}, eval metric: {}\n".format(
                        name, self.nonquant_layer_metrics[name]))
        _logger.info('Analysis file is saved in {}'.format(analysis_file))

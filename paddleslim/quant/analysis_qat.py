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
import paddle.nn.functional as F
from paddle.framework import core
from paddle.fluid.framework import IrGraph
from ..common import get_logger, load_inference_model

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
            qat_metric(float, optional): the metric of the quantized model, which will be calculated automatically if is None
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
        self.data_loader = data_loader
        self.save_dir = save_dir
        self.checkpoint_name = os.path.join(save_dir, 'analysis_checkpoint.pkl')
        self.nonquant_layer_metrics = {}
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        devices = paddle.device.get_device().split(':')[0]
        self.places = paddle.device._convert_to_place(devices)
        executor = paddle.static.Executor(self.places)
        [program, self.feed_list, self.fetch_list] = load_inference_model(
            self.quant_model_dir,
            executor=executor,
            model_filename=self.model_filename,
            params_filename=self.params_filename)
        _logger.info('Loaded model from: {}'.format(quant_model_dir))

        graph = IrGraph(core.Graph(program.desc), for_test=True)

        # find all inputs for each quantizable op
        self.inputs_of_quantized_op = []
        sorted_ops = graph.topology_sort()
        for op_node in sorted_ops:
            op_name = op_node.name()
            if op_name in quantizable_op_type:
                input_names = op_node.op().input_arg_names()
                for input_name in input_names:
                    if 'quantized' in input_name:
                        self.inputs_of_quantized_op.append(input_names)
                        break
        if self.eval_function is None:
            assert self.data_loader is not None, "DataLoader cannot be None if Eval Fuction is None."
            _logger.info(
                'The sensitivity will measured by cosine similarity of the outputs from float model and quantized model.'
            )

        if self.qat_metric is None and self.eval_function is not None:
            _logger.info('Calculating the metric of QAT model...')
            self.qat_metric = self.eval_function(
                executor, program, self.feed_list, self.fetch_list) * 100
            _logger.info('The metric of QAT model is {}'.format(
                round(self.qat_metric, 4)))
        executor.close()

        if resume:
            self.load_checkpoint()

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

    def get_weight_name(self, inputs_names):
        # TODO(xc)
        w_idx = 0 if 'w_0' in inputs_names[0] else 1
        weight_name = inputs_names[w_idx].split('.quantized.dequantized')[0]
        return weight_name

    def get_new_in_out_map(
            self,
            input_list,
            graph,
            float_scope,
            quant_scope, ):

        input_rename_map = {}
        output_rename_map = {}
        removed_ops = []
        for op_node in graph.all_op_nodes():
            if op_node.id() in removed_ops:
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
                            removed_ops.append(op.id())
                            input_rename_map[out_var.node] = in_var_1
                else:
                    # weight
                    with paddle.static.scope_guard(float_scope):
                        float_weight = np.array(
                            float_scope.find_var(in_var.name()).get_tensor())
                    with paddle.static.scope_guard(quant_scope):
                        quant_scope.find_var(in_var.name()).get_tensor().set(
                            float_weight, self.places)
                    input_rename_map[out_var.node] = in_var
                graph.safe_remove_nodes(op_node)
                removed_ops.append(op_node.id())
                output_rename_map[in_var.node] = out_var

        return input_rename_map, output_rename_map, removed_ops

    def relink_graph(self, graph, input_rename_map, output_rename_map,
                     removed_ops):
        for op_node in graph.all_op_nodes():
            if op_node.id() in removed_ops:
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

        return graph.to_program()

    def fp_int_cosine_similarity(self, executor, float_program, quant_program,
                                 float_scope, quant_scope):
        cosine_similarity = []
        for step, data in enumerate(self.data_loader()):
            with paddle.static.scope_guard(float_scope):
                float_preds = executor.run(program=float_program,
                                           feed=data,
                                           fetch_list=self.fetch_list,
                                           return_numpy=False)
                float_preds = float_preds[0]
            with paddle.static.scope_guard(quant_scope):
                quant_preds = executor.run(program=quant_program,
                                           feed=data,
                                           fetch_list=self.fetch_list,
                                           return_numpy=False)
                quant_preds = quant_preds[0]
            paddle.disable_static()
            float_preds = paddle.to_tensor(float_preds)
            quant_preds = paddle.to_tensor(quant_preds)
            cos_sim = F.cosine_similarity(float_preds, quant_preds).mean()
            cos_sim = cos_sim.numpy()
            cosine_similarity.append(cos_sim)
            if step != 0 and (step % 10 == 0):
                _logger.info("[step]: %d, cosine similarity: %.9f" %
                             (step, np.array(cosine_similarity).mean()))
            paddle.enable_static()

        return np.array(cosine_similarity).mean()

    def metric_error_analyse(self):
        executor = paddle.static.Executor(self.places)

        float_scope = paddle.static.Scope()
        quant_scope = paddle.static.Scope()

        for idx, input_list in enumerate(self.inputs_of_quantized_op):
            weight_name = self.get_weight_name(input_list)
            if weight_name in self.nonquant_layer_metrics:
                continue
            _logger.info(
                'Checking {}/{} quant model: without quant layer {}'.format(
                    idx + 1, len(self.inputs_of_quantized_op), weight_name))

            with paddle.static.scope_guard(float_scope):
                [float_program, _, _] = load_inference_model(
                    self.float_model_dir,
                    executor=executor,
                    model_filename=self.model_filename,
                    params_filename=self.params_filename)

            with paddle.static.scope_guard(quant_scope):
                [program, self.feed_list,
                 self.fetch_list] = load_inference_model(
                     self.quant_model_dir,
                     executor=executor,
                     model_filename=self.model_filename,
                     params_filename=self.params_filename)

            program_copy = program.clone()
            graph = IrGraph(core.Graph(program_copy.desc), for_test=True)
            input_rename_map, output_rename_map, removed_ops = self.get_new_in_out_map(
                input_list, graph, float_scope, quant_scope)
            saved_program = self.relink_graph(graph, input_rename_map,
                                              output_rename_map, removed_ops)
            if self.eval_function is not None:
                with paddle.static.scope_guard(quant_scope):
                    _logger.info('Skip quant {}, evaluating....'.format(
                        weight_name))
                    metric = self.eval_function(executor, saved_program,
                                                self.feed_list,
                                                self.fetch_list) * 100
                    self.nonquant_layer_metrics[
                        weight_name] = metric - self.qat_metric
                    _logger.info(
                        'When skip quant %s, the eval metric is %.4f, the sensitive metric is %.4f'
                        % (weight_name, metric, metric - self.qat_metric))
            else:
                metric = self.fp_int_cosine_similarity(executor, float_program,
                                                       saved_program,
                                                       float_scope, quant_scope)
                self.nonquant_layer_metrics[weight_name] = 1 - metric
                _logger.info(
                    'When skip quant %s, the cosine similarity is %.4f, the sensitive metric is %.4f'
                    % (weight_name, metric, 1 - metric))
            self.save_checkpoint()

        executor.close()

        self.sensitivity_ranklist = sorted(
            self.nonquant_layer_metrics,
            key=self.nonquant_layer_metrics.get,
            reverse=True)
        _logger.info('Finished computing the sensitivity of the model.')
        for name in self.sensitivity_ranklist:
            _logger.info("Without quant layer name: {}, sensitive metric: {}".
                         format(name, self.nonquant_layer_metrics[name]))

        analysis_file = os.path.join(self.save_dir, "analysis.txt")
        with open(analysis_file, "w") as analysis_ret_f:
            for name in self.sensitivity_ranklist:
                analysis_ret_f.write(
                    "Without quant layer name: {}, sensitive metric: {}\n".
                    format(name, self.nonquant_layer_metrics[name]))
        _logger.info('Analysis file is saved in {}'.format(analysis_file))

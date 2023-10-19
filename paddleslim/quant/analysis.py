# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import csv
import numpy as np
import random
import tempfile
import paddle
from ..common import get_logger, load_inference_model
from paddle.framework import IrGraph
from paddle.framework import core

from paddle.static.quantization import PostTrainingQuantization
from .analysis_utils import *
_logger = get_logger(__name__, level=logging.INFO)

SUPPORT_WEIGHT_OP_DICT = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
    "conv2d_transpose": [["Input", "Filter"], ["Output"]],
    "mul": [["X", "Y"], ["Out"]],
    "matmul": [["X", "Y"], ["Out"]],
    "matmul_v2": [["X", "Y"], ["Out"]]
}


class Analysis(object):
    def __init__(self,
                 float_model_dir,
                 quant_model_dir=None,
                 model_filename=None,
                 params_filename=None,
                 data_loader=None,
                 eval_function=None,
                 resume=False,
                 save_dir='analysis_results',
                 quant_config=None):
        '''
        Analysis provides to analysis the sensitivity of each op in the model.
        
        Args:
            float_model_dir(str, required): the path of fp32 model, it can also be '.onnx'
            quant_model_dir(str, optional):the path of quantized model, if is None, float model will be quantized by PTQ
            model_filename(str, optional): the model file name of the fp32 and quantized model
            params_filename(str, optional): the parameter file name of the fp32 and quantized model
            eval_function(function): eval function, define by yourself to return the metric of the inference program, can be used to judge the metric of quantized model.  (TODO: optional)
            data_loader(Python Generator, Paddle.io.DataLoader, optional): the
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time
            save_dir(str, optional): the output dir that stores the analyzed information
            resume(bool, optional): When break off while ananlyzing, could resume analysis program and load already analyzed information.
            quant_config(dict, optional): the args that can initialize PostTrainingQuantization
        
        Examples:
        .. code-block:: python
       
        from paddleslim.quant.analysis import Analysis
        analyzer = Analysis(quant_model_dir=quant_model_dir)
        analyzer.metric_error_analyse()
          
        '''
        if model_filename is None:
            model_filename = 'model.pdmodel'
        if params_filename is None:
            params_filename = 'model.pdiparams'
        self.float_model_dir = float_model_dir
        self.quant_model_dir = quant_model_dir
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.histogram_bins = 1000
        self.save_dir = save_dir
        self.checkpoint_name = os.path.join(save_dir, 'analysis_checkpoint.pkl')
        self.data_loader = data_loader
        self.eval_function = eval_function
        self.quant_config = quant_config
        self.batch_nums = quant_config.get("batch_nums", 10)
        self.is_full_quantize = quant_config.get("is_full_quantize", False)
        self.onnx_format = quant_config.get("onnx_format", False)

        self.quantizable_op_type = quant_config.get(
            "quantizable_op_type", list(SUPPORT_WEIGHT_OP_DICT.keys()))
        self.skip_tensor_list = quant_config.get("skip_tensor_list", [])
        if self.skip_tensor_list:
            del self.quant_config['skip_tensor_list']
        quant_config['onnx_format'] = self.onnx_format
        quant_config['algo'] = quant_config.get("algo", 'avg')

        if self.onnx_format:
            self.temp_root_path = tempfile.TemporaryDirectory(dir=self.save_dir)
            self.temp_save_path = os.path.join(self.temp_root_path.name, "ptq")
            if not os.path.exists(self.temp_save_path):
                os.makedirs(self.temp_save_path)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        devices = paddle.device.get_device().split(':')[0]
        self.places = paddle.device._convert_to_place(devices)

        self.layer_metrics = {}
        if resume:
            self.load_checkpoint()

    def save_checkpoint(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.checkpoint_name, 'wb') as f:
            pickle.dump(self.layer_metrics, f)
        _logger.info('Save checkpoint to {}.'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            _logger.info('Checkpoint path {} does not exist.'.format(
                self.checkpoint_name))
            return False
        with open(self.checkpoint_name, 'rb') as f:
            self.layer_metrics = pickle.load(f)
        _logger.info('Load checkpoint from {}.'.format(self.checkpoint_name))
        return True

    def get_weight_act_info(self, program, persistable=True):
        self.persistable_var_names = []
        for var in program.list_vars():
            if var.persistable:
                self.persistable_var_names.append(var.name)
        graph = IrGraph(core.Graph(program.desc), for_test=True)

        weight_act_dict = {}
        act_weight_dict = {}
        ops = graph.all_op_nodes()
        for op_node in ops:
            if op_node.name() in self.quantizable_op_type:
                in_x, in_y = SUPPORT_WEIGHT_OP_DICT[op_node.name()][0]
                input_name_x = op_node.input(in_x)[0]
                input_name_y = op_node.input(in_y)[0]
                if not persistable:
                    weight_act_dict[input_name_y] = input_name_x
                    act_weight_dict[input_name_x] = input_name_y
                else:
                    if input_name_y in self.persistable_var_names and input_name_y not in self.skip_tensor_list:
                        weight_act_dict[input_name_y] = input_name_x
                        act_weight_dict[input_name_x] = input_name_y
        return weight_act_dict, act_weight_dict

    def create_ptq(self, executor, skip_tensor_list=[]):
        skip_tensor_list += self.skip_tensor_list
        return PostTrainingQuantization(
            executor=executor,
            data_loader=self.data_loader,
            model_dir=self.float_model_dir,
            model_filename=self.model_filename,
            params_filename=self.params_filename,
            skip_tensor_list=skip_tensor_list,
            **self.quant_config)

    def sampling(self, executor, program, scope, fetch_list):
        batch_id = 0
        for data in self.data_loader():
            executor.run(
                program=program,
                feed=data,
                fetch_list=fetch_list,
                return_numpy=False,
                scope=scope)
            batch_id += 1
            if batch_id >= self.batch_nums:
                break

    def collect_base_stat(self):
        _logger.info('Collecting fp model statistic...')

        executor = paddle.static.Executor(self.places)
        [program, feed_list, fetch_list]= load_inference_model( \
            self.float_model_dir, \
            executor=executor, \
            model_filename=self.model_filename, \
            params_filename=self.params_filename)
        scope = paddle.static.global_scope()

        self.fp_weight_act_dict, self.fp_act_weight_dict = self.get_weight_act_info(
            program)
        self.fp_weight_names = list(self.fp_weight_act_dict.keys())
        self.fp_act_names = list(self.fp_weight_act_dict.values())

        for var in program.list_vars():
            if var.name in self.fp_act_names:
                var.persistable = True

        # sample
        self.sampling(executor, program, scope, fetch_list)
        fp_act = collect_vars(scope, self.fp_act_names)
        fp_weight = collect_vars(scope, self.fp_weight_names)
        executor.close()
        return fp_act, fp_weight

    def collect_quant_stat(self):
        _logger.info('Collecting quant model statistic...')
        if self.quant_model_dir is None:
            executor = paddle.static.Executor(self.places)
            scope = paddle.static.global_scope()
            ptq = self.create_ptq(executor)
            program = ptq.quantize()
            feed_list, fetch_list = ptq._feed_list, ptq._fetch_list
        else:
            executor = paddle.static.Executor(self.places)
            [program, feed_list, fetch_list]= load_inference_model( \
                self.quant_model_dir, \
                executor=executor, \
                model_filename=self.model_filename, \
                params_filename=self.params_filename)
            scope = paddle.static.global_scope()

        self.quant_weight_act_dict, self.quant_act_weight_dict = self.get_weight_act_info(
            program)
        self.quant_weight_names = list(self.quant_weight_act_dict.keys())
        self.quant_act_names = list(self.quant_weight_act_dict.values())

        for var in program.list_vars():
            if var.name in self.quant_act_names:
                var.persistable = True

        self.sampling(executor, program, scope, fetch_list)

        quant_act = collect_vars(scope, self.quant_act_names)
        quant_weight = collect_vars(scope, self.quant_weight_names)
        executor.close()
        return quant_act, quant_weight

    def collect_statistic(self,
                          fp_tensors,
                          quant_tensors,
                          var_name_map,
                          is_weight,
                          axis=None):
        statistic = []
        box_fp_dist, box_q_dist = {}, {}
        hist_fp_dist, hist_q_dist = {}, {}
        fp_tensor_names = sorted(list(fp_tensors.keys()))
        for var_name in fp_tensor_names:
            fp_tensor = fp_tensors[var_name]
            quant_name = var_name_map[
                var_name] if var_name_map is not None else var_name
            quant_tensor = quant_tensors[quant_name]
            diff = fp_tensor - quant_tensor

            fp_min = round(fp_tensor.min(), 4)
            fp_max = round(fp_tensor.max(), 4)
            fp_mean = round(fp_tensor.mean(), 4)
            fp_std = round(fp_tensor.std(), 4)

            q_min = round(quant_tensor.min(), 4)
            q_max = round(quant_tensor.max(), 4)
            q_mean = round(quant_tensor.mean(), 4)
            q_std = round(quant_tensor.std(), 4)

            diff_min = round(diff.min(), 4)
            diff_max = round(diff.max(), 4)
            diff_mean = round(diff.mean(), 4)
            diff_std = round(diff.std(), 4)

            stat = {
                'Var Name':
                var_name,
                'Var Type':
                'Weight' if is_weight else 'Activation',
                'Corresponding Weight Name':
                self.fp_act_weight_dict[var_name] if not is_weight else None,
                'FP32 Min':
                fp_min,
                'FP32 Max':
                fp_max,
                'FP32 Mean':
                fp_mean,
                'FP32 Std':
                fp_std,
                'Quantized Min':
                q_min,
                'Quantized Max':
                q_max,
                'Quantized Mean':
                q_mean,
                'Quantized Std':
                q_std,
                'Diff Min':
                diff_min,
                'Diff Max':
                diff_max,
                'Diff Mean':
                diff_mean,
                'Diff Std':
                diff_std,
            }
            statistic.append(stat)
            # for boxplot
            if axis is None:
                box_fp_tensor = fp_tensor.flatten()
                box_q_tensor = quant_tensor.flatten()
            else:
                box_fp_tensor = fp_tensor.reshape(
                    [-1, fp_tensor.shape[axis]]).abs().max(axis=-1)
                box_q_tensor = quant_tensor.reshape(
                    [-1, quant_tensor.shape[axis]]).abs().max(axis=-1)
            sample_num = len(
                box_fp_tensor) if len(box_fp_tensor) < 1000 else 1000
            box_fp_tensor = random.sample(list(box_fp_tensor), sample_num)
            box_q_tensor = random.sample(list(box_q_tensor), sample_num)
            box_fp_dist[var_name] = box_fp_tensor
            box_q_dist[quant_name] = box_q_tensor

            # for histplot
            _, hist_edges = np.histogram(
                fp_tensor.copy(), bins=50, range=(fp_min, fp_max))
            hist_fp_dist[var_name] = [fp_tensor.flatten(), hist_edges]
            _, hist_edges = np.histogram(
                quant_tensor.copy(), bins=50, range=(q_min, q_max))
            hist_q_dist[quant_name] = [quant_tensor.flatten(), hist_edges]

        return statistic, box_fp_dist, box_q_dist, hist_fp_dist, hist_q_dist

    def statistical_analyse(self, analysis_axis=None):
        fp_act, fp_weight = self.collect_base_stat()
        quant_act, quant_weight = self.collect_quant_stat()
        fp_q_act_dict = {
            self.fp_weight_act_dict[n]: self.quant_weight_act_dict[n]
            for n in self.fp_weight_act_dict
        }
        act_statistic, box_fp_dist, box_q_dist, hist_fp_dist, hist_q_dist = self.collect_statistic(
            fp_act,
            quant_act,
            fp_q_act_dict,
            is_weight=False,
            axis=analysis_axis)

        plot_box_distribution(box_fp_dist, self.save_dir,
                              'fp_activation_boxplot.pdf')
        plot_box_distribution(box_q_dist, self.save_dir,
                              'quantized_activation_boxplot.pdf')
        plot_hist_distribution(hist_fp_dist, self.save_dir,
                               'fp_activation_histplot.pdf')
        plot_hist_distribution(hist_q_dist, self.save_dir,
                               'quantized_activation_histplot.pdf')

        weight_statistic, box_fp_dist, box_q_dist, hist_fp_dist, hist_q_dist = self.collect_statistic(
            fp_weight, quant_weight, None, is_weight=True, axis=analysis_axis)
        plot_box_distribution(box_fp_dist, self.save_dir,
                              'fp_weight_boxplot.pdf')
        plot_box_distribution(box_q_dist, self.save_dir,
                              'quantized_weight_boxplot.pdf')
        plot_hist_distribution(hist_fp_dist, self.save_dir,
                               'fp_weight_histplot.pdf')
        plot_hist_distribution(hist_q_dist, self.save_dir,
                               'quantized_weight_histplot.pdf')

        statistic = act_statistic + weight_statistic
        csv_columns = [
            'Var Name', 'Var Type', 'Corresponding Weight Name', 'FP32 Min',
            'FP32 Max', 'FP32 Mean', 'FP32 Std', 'Quantized Min',
            'Quantized Max', 'Quantized Mean', 'Quantized Std', 'Diff Min',
            'Diff Max', 'Diff Mean', 'Diff Std'
        ]
        save_csv(statistic, self.save_dir, 'statistic.csv', csv_columns)

    def get_quant_sensitive_metric(self, skip_list, layer_name):
        executor = paddle.static.Executor(self.places)
        if self.eval_function is not None:
            ptq = self.create_ptq(executor, skip_list)
            program = ptq.quantize()
            _logger.info('Evaluating...')
            if self.onnx_format:
                post_training_quantization.save_quantized_model(
                    self.temp_save_path,
                    model_filename='model.pdmodel',
                    params_filename='model.pdiparams')
                program, feed_list, fetch_list = load_inference_model(
                    self.temp_save_path,
                    executor,
                    model_filename='model.pdmodel',
                    params_filename='model.pdiparams')
            metric = self.eval_function(executor, program, ptq._feed_list,
                                        ptq._fetch_list)
            sensitive_metric = self.fp_metric - metric
            _logger.info(
                "Quantized layer name: %s, the accuracy: %.4f, the sensitive metric: %.4f"
                % (layer_name, metric, sensitive_metric))
        else:
            float_scope = paddle.static.Scope()
            quant_scope = paddle.static.Scope()
            with paddle.static.scope_guard(float_scope):
                [float_program, float_feed_list,
                 float_fetch_list] = load_inference_model(
                     self.float_model_dir,
                     executor=executor,
                     model_filename=self.model_filename,
                     params_filename=self.params_filename)

            with paddle.static.scope_guard(quant_scope):
                ptq = self.create_ptq(executor, skip_list)
                quant_program = ptq.quantize()

            metric = fp_quant_cosine_similarity(
                executor, self.data_loader, float_program, quant_program,
                float_scope, quant_scope, float_fetch_list, ptq._fetch_list)
            sensitive_metric = 1.0 - metric
            _logger.info(
                "Quantized layer name: %s, the cosine similarity: %.4f, the sensitive metric: %.4f"
                % (layer_name, metric, sensitive_metric))

        executor.close()
        return sensitive_metric

    def get_dequant_sensitive_metric(self, executor, float_scope, quant_scope,
                                     layer_name):
        weight_name = layer_name.split('.quantized.dequantized')[0]
        with paddle.static.scope_guard(float_scope):
            [float_program, float_feed_list,
             float_fetch_list] = load_inference_model(
                 self.float_model_dir,
                 executor=executor,
                 model_filename=self.model_filename,
                 params_filename=self.params_filename)

        with paddle.static.scope_guard(quant_scope):
            [program, quant_feed_list, quant_fetch_list] = load_inference_model(
                self.quant_model_dir,
                executor=executor,
                model_filename=self.model_filename,
                params_filename=self.params_filename)

        program_copy = program.clone()
        graph = IrGraph(core.Graph(program_copy.desc), for_test=True)
        input_rename_map, output_rename_map, removed_ops = get_new_in_out_map(
            self.weight_act_dict[layer_name], graph, float_scope, quant_scope,
            self.places)
        saved_program = relink_graph(graph, input_rename_map, output_rename_map,
                                     removed_ops)
        if self.eval_function is not None:
            with paddle.static.scope_guard(quant_scope):
                _logger.info(
                    'Skip quant {}, evaluating....'.format(weight_name))
                metric = self.eval_function(executor, saved_program,
                                            quant_feed_list, quant_fetch_list)
                sensitive_metric = self.quant_metric - metric
                _logger.info(
                    'When skip quant %s, the eval metric is %.4f, the sensitive metric is %.4f'
                    % (weight_name, metric, self.quant_metric - metric))
        else:
            metric = fp_quant_cosine_similarity(
                executor, self.data_loader, float_program, saved_program,
                float_scope, quant_scope, float_fetch_list, quant_fetch_list)
            sensitive_metric = 1 - metric
            _logger.info(
                'When skip quant %s, the cosine similarity is %.4f, the sensitive metric is %.4f'
                % (weight_name, metric, 1 - metric))
        return sensitive_metric

    def prepare_error_analyse(self, dequant_layer_by_layer):
        if not dequant_layer_by_layer:
            executor = paddle.static.Executor(self.places)
            [program, feed_list, fetch_list]= load_inference_model( \
                self.float_model_dir, \
                executor=executor, \
                model_filename=self.model_filename, \
                params_filename=self.params_filename)

            self.weight_act_dict, _ = self.get_weight_act_info(program)
            self.support_quant_name_list = list(self.weight_act_dict.keys())
            self.tobe_analyized_layer = sorted(
                list(
                    set(self.support_quant_name_list) -
                    set(self.skip_tensor_list)))
            if self.eval_function is not None:
                _logger.info('Start to evaluate the FP model.')
                self.fp_metric = self.eval_function(executor, program,
                                                    feed_list, fetch_list)
                _logger.info(
                    'The accuracy of the FP model is: %.4f' % self.fp_metric)
                executor.close()

                _logger.info('Start to evaluate the quantized model.')
                executor = paddle.static.Executor(self.places)
                ptq = self.create_ptq(executor, self.skip_tensor_list)
                program = ptq.quantize()
                self.quant_metric = self.eval_function(executor, program,
                                                       feed_list, fetch_list)
                _logger.info('The accuracy of the quantized model is: %.4f' %
                             self.quant_metric)
        else:
            executor = paddle.static.Executor(self.places)
            [program, feed_list, fetch_list] = load_inference_model(
                self.quant_model_dir,
                executor=executor,
                model_filename=self.model_filename,
                params_filename=self.params_filename)
            graph = IrGraph(core.Graph(program.desc), for_test=True)

            self.weight_act_dict, _ = self.get_weight_act_info(
                program, persistable=False)

            if self.eval_function is not None:
                _logger.info('Start to evaluate the quantized model.')
                self.quant_metric = self.eval_function(executor, program,
                                                       feed_list, fetch_list)
                _logger.info('The accuracy of the quantized model is: %.4f' %
                             self.quant_metric)
            executor.close()

    def metric_error_analyse(self):
        assert self.data_loader is not None, \
        "When computing the sensitivity of quantized layers, the data loader is needed"
        dequant_layer_by_layer = False if self.quant_model_dir is None else True
        self.prepare_error_analyse(dequant_layer_by_layer)
        if not dequant_layer_by_layer:
            _logger.info(
                'For each layer, quantize the weight op and evaluate the quantized model.'
            )
            # For each layer, quantize the weight op and evaluate the quantized model.
            for i, layer_name in enumerate(self.tobe_analyized_layer):
                if layer_name in self.layer_metrics:
                    continue
                _logger.info(
                    'Checking {}/{} quant model: quant layer {}'.format(
                        i + 1, len(self.tobe_analyized_layer), layer_name))
                skip_list = copy.copy(list(self.support_quant_name_list))
                skip_list.remove(layer_name)
                sensitive_metric = self.get_quant_sensitive_metric(
                    skip_list, layer_name)
                self.layer_metrics[layer_name] = sensitive_metric
                self.save_checkpoint()

            if self.onnx_format:
                self.temp_root_path.cleanup()

        else:
            _logger.info(
                'For each layer, dequantize the weight op and evaluate the quantized model.'
            )
            executor = paddle.static.Executor(self.places)
            float_scope = paddle.static.Scope()
            quant_scope = paddle.static.Scope()
            for idx, name in enumerate(self.weight_act_dict):
                weight_name = name.split('.quantized.dequantized')[0]
                if weight_name in self.layer_metrics:
                    continue
                _logger.info(
                    'Checking {}/{} quant model: without quant layer {}'.format(
                        idx + 1, len(self.weight_act_dict), weight_name))
                sensitive_metric = self.get_dequant_sensitive_metric(
                    executor, float_scope, quant_scope, name)
                self.layer_metrics[weight_name] = sensitive_metric
                self.save_checkpoint()
            executor.close()

        self.sensitivity_ranklist = sorted(
            self.layer_metrics, key=self.layer_metrics.get, reverse=True)

        _logger.info('Finished computing the sensitivity of the model.')
        for name in self.sensitivity_ranklist:
            _logger.info("layer name: {}, sensitivity metric: {}".format(
                name, self.layer_metrics[name]))

        analysis_file = os.path.join(self.save_dir, "analysis.txt")
        with open(analysis_file, "w") as analysis_ret_f:
            for name in self.sensitivity_ranklist:
                analysis_ret_f.write("layer name: {}, sensitivity metric: {}\n".
                                     format(name, self.layer_metrics[name]))
        _logger.info('Analysis file is saved in {}'.format(analysis_file))

    def get_target_quant_model(self, target_metric):
        _logger.info(
            'Start to Find quantized model that satisfies the target metric.')
        _logger.info(
            'Make sure that you are using full eval dataset to get target quantized model.'
        )
        skip_list = []
        if self.layer_metrics:
            rank_list = sorted(
                self.layer_metrics, key=self.layer_metrics.get, reverse=True)
        else:
            _logger.info(
                'Analyse metric error before get target quantized model.')
            self.metric_error_analyse()

        while len(rank_list) > 0:
            skip_list.append(rank_list.pop(0))
            _logger.info('Skip Ops: {}'.format(skip_list))
            executor = paddle.static.Executor(self.places)
            ptq = self.create_ptq(executor, skip_list)
            program = ptq.quantize()

            _logger.info('Evaluating...')
            quant_metric = self.eval_function(executor, program, ptq._feed_list,
                                              ptq._fetch_list)
            _logger.info("Current eval metric: {}, the target metric: {}".
                         format(quant_metric, target_metric))
            if quant_metric >= target_metric:
                quantize_model_path = os.path.join(self.save_dir,
                                                   'target_quant_model')
                _logger.info(
                    'The quantized model satisfies the target metric and is saved to {}'.
                    format(quantize_model_path))
                ptq.save_quantized_model(
                    quantize_model_path,
                    model_filename='model.pdmodel',
                    params_filename='model.pdiparams')
                break
            else:
                _logger.info(
                    'The quantized model does not satisfy the target metric. Skip next Op...'
                )
            executor.close()
        else:
            _logger.info(
                'Sorry, the target quantized model cannot be found. Please set lower target metric.'
            )

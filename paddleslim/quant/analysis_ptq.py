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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import numpy as np
import random
import tempfile
import paddle
import paddle.nn.functional as F
from ..core import GraphWrapper
from ..common import get_logger
from ..common import get_feed_vars, wrap_dataloader, load_inference_model, get_model_dir

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["AnalysisPTQ"]


class AnalysisPTQ(object):
    def __init__(self,
                 model_dir,
                 model_filename=None,
                 params_filename=None,
                 eval_function=None,
                 data_loader=None,
                 save_dir='analysis_results',
                 resume=False,
                 ptq_config=None):
        """
        AnalysisPTQ provides to analysis the sensitivity of each op in the model.
        
        Args:
            model_dir(str): the path of fp32 model that will be quantized, it can also be '.onnx'
            model_filename(str, optional): the model file name of the fp32 model
            params_filename(str, optional): the parameter file name of the fp32 model
            eval_function(function): eval function, define by yourself to return the metric of the inference program, can be used to judge the metric of quantized model.  (TODO: optional)
            data_loader(Python Generator, Paddle.io.DataLoader, optional): the
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time
            save_dir(str, optional): the output dir that stores the analyzed information
            resume(bool, optional): When break off while ananlyzing, could resume analysis program and load already analyzed information.
            ptq_config(dict, optional): the args that can initialize PostTrainingQuantization
            
        """
        if model_filename is None:
            model_filename = 'model.pdmodel'
        if params_filename is None:
            params_filename = 'model.pdiparams'
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.histogram_bins = 1000
        self.save_dir = save_dir
        self.eval_function = eval_function
        self.quant_layer_names = []
        self.checkpoint_name = os.path.join(save_dir, 'analysis_checkpoint.pkl')
        self.quant_layer_metrics = {}
        self.ptq_config = ptq_config
        self.batch_nums = ptq_config[
            'batch_nums'] if 'batch_nums' in ptq_config else 10
        self.is_full_quantize = ptq_config[
            'is_full_quantize'] if 'is_full_quantize' in ptq_config else False
        self.onnx_format = ptq_config[
            'onnx_format'] if 'onnx_format' in ptq_config else False
        ptq_config['onnx_format'] = self.onnx_format
        if 'algo' not in ptq_config:
            ptq_config['algo'] = 'avg'

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if self.onnx_format:
            self.temp_root_path = tempfile.TemporaryDirectory(dir=self.save_dir)
            self.temp_save_path = os.path.join(self.temp_root_path.name, "ptq")
            if not os.path.exists(self.temp_save_path):
                os.makedirs(self.temp_save_path)

        devices = paddle.device.get_device().split(':')[0]
        self.places = paddle.device._convert_to_place(devices)
        executor = paddle.static.Executor(self.places)

        # load model 
        [program, self.feed_list, self.fetch_list]= load_inference_model( \
            self.model_dir, \
            executor=executor, \
            model_filename=self.model_filename, \
            params_filename=self.params_filename)

        # create data_loader
        self.data_loader = wrap_dataloader(data_loader, self.feed_list)

        # quant model to get quantizable ops 
        post_training_quantization = self.create_ptq(executor, None)

        _logger.info('Run PTQ before analysis.')
        program = post_training_quantization.quantize()

        if self.onnx_format:
            post_training_quantization.save_quantized_model(
                self.temp_save_path,
                model_filename='model.pdmodel',
                params_filename='model.pdiparams')
            program, _, _ = load_inference_model(
                self.temp_save_path,
                executor,
                model_filename='model.pdmodel',
                params_filename='model.pdiparams')

        # get quantized weight and act var name
        self.quantized_weight_var_name = post_training_quantization._quantized_weight_var_name
        self.quantized_act_var_name = post_training_quantization._quantized_act_var_name
        self.support_quant_val_name_list = self.quantized_weight_var_name if not self.is_full_quantize else list(
            self.quantized_act_var_name)
        self.weight_names = list(self.quantized_weight_var_name)
        self.act_names = list(self.quantized_act_var_name)
        executor.close()

        # load tobe_analyized_layer from checkpoint
        if resume:
            self.load_checkpoint()
        self.tobe_analyized_layer = sorted(
            list(self.support_quant_val_name_list))

    def save_checkpoint(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.checkpoint_name, 'wb') as f:
            pickle.dump(self.quant_layer_metrics, f)
        _logger.info('Save checkpoint to {}.'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            _logger.info('Checkpoint path {} does not exist.'.format(
                self.checkpoint_name))
            return False
        with open(self.checkpoint_name, 'rb') as f:
            self.quant_layer_metrics = pickle.load(f)
        _logger.info('Load checkpoint from {}.'.format(self.checkpoint_name))
        return True

    def save_csv(self, data, save_name, csv_columns):
        save_path = os.path.join(self.save_dir, save_name)
        with open(save_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for d in data:
                writer.writerow(d)
        _logger.info('Activation Statistic is saved in {}'.format(save_path))

    def create_ptq(self, executor, skip_tensor_list):
        return paddle.fluid.contrib.slim.quantization.PostTrainingQuantization(
            executor=executor,
            data_loader=self.data_loader,
            model_dir=self.model_dir,
            model_filename=self.model_filename,
            params_filename=self.params_filename,
            skip_tensor_list=skip_tensor_list,
            **self.ptq_config)

    def sampling(self, executor, program, scope):
        batch_id = 0
        for data in self.data_loader():
            executor.run(program=program,
                         feed=data,
                         fetch_list=self.fetch_list,
                         return_numpy=False,
                         scope=scope)
            batch_id += 1
            if batch_id >= self.batch_nums:
                break

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

    def get_sensitive_metric(self, skip_list, layer_name):
        executor = paddle.static.Executor(self.places)
        if self.eval_function is not None:
            post_training_quantization = self.create_ptq(executor, skip_list)
            program = post_training_quantization.quantize()
            _logger.info('Evaluating...')
            if self.onnx_format:
                post_training_quantization.save_quantized_model(
                    self.temp_save_path,
                    model_filename='model.pdmodel',
                    params_filename='model.pdiparams')
                program, _, _ = load_inference_model(
                    self.temp_save_path,
                    executor,
                    model_filename='model.pdmodel',
                    params_filename='model.pdiparams')
            metric = self.eval_function(executor, program, self.feed_list,
                                        self.fetch_list)
            if skip_list is None:
                executor.close()
                return metric

            sensitive_metric = self.base_metric - metric
            _logger.info(
                "Quantized layer name: %s, the accuracy: %.4f, the sensitive metric: %.4f"
                % (layer_name, metric, sensitive_metric))
        else:
            float_scope = paddle.static.Scope()
            quant_scope = paddle.static.Scope()
            with paddle.static.scope_guard(float_scope):
                [float_program, _, _] = load_inference_model(
                    self.model_dir,
                    executor=executor,
                    model_filename=self.model_filename,
                    params_filename=self.params_filename)

            with paddle.static.scope_guard(quant_scope):
                post_training_quantization = self.create_ptq(executor,
                                                             skip_list)
                quant_program = post_training_quantization.quantize()

            metric = self.fp_int_cosine_similarity(executor, float_program,
                                                   quant_program, float_scope,
                                                   quant_scope)
            sensitive_metric = 1.0 - metric
            _logger.info(
                "Quantized layer name: %s, the cosine similarity: %.4f, the sensitive metric: %.4f"
                % (layer_name, metric, sensitive_metric))

        executor.close()
        return sensitive_metric

    def metric_error_analyse(self):
        '''
        Evaluate the quantized models, which are generated by quantizing each weight operator one by one. The results will be saved into analysis.txt.
        '''
        assert self.data_loader is not None, "When computing the sensitivity of quantized layers, the data loader is needed"
        if self.eval_function is not None:
            # evaluate before quant 
            _logger.info('Start to evaluate the base model.')
            executor = paddle.static.Executor(self.places)
            [program, feed_list, fetch_list]= load_inference_model( \
                self.model_dir, \
                executor=executor, \
                model_filename=self.model_filename, \
                params_filename=self.params_filename)
            self.base_metric = self.eval_function(executor, program, feed_list,
                                                  fetch_list)
            _logger.info('Before quantized, the accuracy of the model is: {}'.
                         format(self.base_metric))
            executor.close()

            # evaluate before quant 
            _logger.info('Start to evaluate the quantized model.')
            self.quant_metric = self.get_sensitive_metric(
                None, 'all quantizable layers')
            _logger.info('After quantized, the accuracy of the model is: {}'.
                         format(self.quant_metric))

        # For each layer, quantize the weight op and evaluate the quantized model.
        for i, layer_name in enumerate(self.tobe_analyized_layer):
            if layer_name in self.quant_layer_metrics:
                continue

            _logger.info('Checking {}/{} quant model: quant layer {}'.format(
                i + 1, len(self.tobe_analyized_layer), layer_name))
            skip_list = copy.copy(list(self.support_quant_val_name_list))
            skip_list.remove(layer_name)
            sensitive_metric = self.get_sensitive_metric(skip_list, layer_name)
            self.quant_layer_metrics[layer_name] = sensitive_metric
            self.save_checkpoint()

        if self.onnx_format:
            self.temp_root_path.cleanup()

        self.sensitivity_ranklist = sorted(
            self.quant_layer_metrics,
            key=self.quant_layer_metrics.get,
            reverse=True)

        _logger.info('Finished computing the sensitivity of the model.')
        for name in self.sensitivity_ranklist:
            _logger.info("Quantized layer name: {}, sensitivity metric: {}".
                         format(name, self.quant_layer_metrics[name]))

        analysis_file = os.path.join(self.save_dir, "analysis.txt")
        with open(analysis_file, "w") as analysis_ret_f:
            for name in self.sensitivity_ranklist:
                analysis_ret_f.write(
                    "Quantized layer name: {}, sensitivity metric: {}\n".format(
                        name, self.quant_layer_metrics[name]))
        _logger.info('Analysis file is saved in {}'.format(analysis_file))

    def collect_vars(self, scope, var_names):
        all_vars = {}
        for var_name in var_names:
            var_tensor = paddle.fluid.contrib.slim.quantization.utils.load_variable_data(
                scope, var_name)
            all_vars[var_name] = var_tensor
        return all_vars

    def collect_base_stat(self):
        _logger.info('Collecting Statistic Before PTQ...')
        executor = paddle.static.Executor(self.places)
        [program, feed_list, fetch_list]= load_inference_model( \
            self.model_dir, \
            executor=executor, \
            model_filename=self.model_filename, \
            params_filename=self.params_filename)
        scope = paddle.static.global_scope()
        persistable_var_names = []
        for var in program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)

        self.acts_weight_map = self.get_weight_act_map(
            program, self.weight_names, persistable_var_names)
        activations_names = list(self.acts_weight_map.keys())
        for var in program.list_vars():
            if var.name in activations_names:
                var.persistable = True

        # sample 
        self.sampling(executor, program, scope)
        before_act_data = self.collect_vars(scope, activations_names)
        before_weight_data = self.collect_vars(scope, self.weight_names)
        executor.close()
        return before_act_data, before_weight_data

    def collect_quant_stat(self):
        _logger.info('Collecting Statistic After PTQ...')
        executor = paddle.static.Executor(self.places)
        scope = paddle.static.global_scope()
        post_training_quantization = self.create_ptq(executor, None)
        program = post_training_quantization.quantize()

        persistable_var_names = []
        for var in program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)

        quant_weight_names = self.weight_names
        dequant_act_names = ["%s.quantized" % (n) for n in self.acts_weight_map]
        for var in program.list_vars():
            if var.name in dequant_act_names:
                var.persistable = True

        self.sampling(executor, program, scope)

        after_act_data = self.collect_vars(scope, dequant_act_names)
        after_weight_data = self.collect_vars(scope, quant_weight_names)
        executor.close()
        return after_act_data, after_weight_data

    def statistical_analyse(self, analysis_axis=None):

        self.act_data, self.weight_data = self.collect_base_stat()
        self.quant_act_data, self.dequant_weight_data = self.collect_quant_stat(
        )

        fp_q_act_name_map = {
            n: "%s.quantized" % (n)
            for n in self.acts_weight_map
        }
        act_statistic, box_fp_dist, box_q_dist, hist_fp_dist, hist_q_dist = self.collect_statistic(
            self.act_data,
            self.quant_act_data,
            fp_q_act_name_map,
            is_weight=False,
            axis=analysis_axis)

        self.plot_box_distribution(box_fp_dist,
                                   list(self.acts_weight_map.keys()),
                                   'fp_activation_boxplot.pdf')
        self.plot_box_distribution(box_q_dist,
                                   list(self.acts_weight_map.keys()),
                                   'quantized_activation_boxplot.pdf')
        self.plot_hist_distribution(hist_fp_dist, 'fp_activation_histplot.pdf')
        self.plot_hist_distribution(hist_q_dist,
                                    'quantized_activation_histplot.pdf')

        weight_statistic, box_fp_dist, box_q_dist, hist_fp_dist, hist_q_dist = self.collect_statistic(
            self.weight_data,
            self.dequant_weight_data,
            None,
            is_weight=True,
            axis=analysis_axis)
        self.plot_box_distribution(box_fp_dist,
                                   list(self.quantized_weight_var_name),
                                   'fp_weight_boxplot.pdf')
        self.plot_box_distribution(box_q_dist,
                                   list(self.quantized_weight_var_name),
                                   'quantized_weight_boxplot.pdf')
        self.plot_hist_distribution(hist_fp_dist, 'fp_weight_histplot.pdf')
        self.plot_hist_distribution(hist_q_dist,
                                    'quantized_weight_histplot.pdf')

        statistic = act_statistic + weight_statistic
        csv_columns = [
            'Var Name', 'Var Type', 'Corresponding Weight Name', 'FP32 Min',
            'FP32 Max', 'FP32 Mean', 'FP32 Std', 'Quantized Min',
            'Quantized Max', 'Quantized Mean', 'Quantized Std', 'Diff Min',
            'Diff Max', 'Diff Mean', 'Diff Std'
        ]
        self.save_csv(statistic, 'statistic.csv', csv_columns)

    def get_weight_act_map(self, program, weight_names, persistable_var_names):
        weight_act_map = {}
        for op_name in weight_names:
            for block_id in range(len(program.blocks)):
                for op in program.blocks[block_id].ops:
                    var_name_list = paddle.fluid.contrib.slim.quantization.utils._get_op_input_var_names(
                        op)
                    if op_name in var_name_list:
                        for var_name in var_name_list:
                            if var_name not in persistable_var_names:
                                weight_act_map[var_name] = op_name
        return weight_act_map

    def collect_statistic(self,
                          fp_tensors,
                          quant_tensors,
                          var_name_map,
                          is_weight,
                          axis=None):
        statistic = []
        box_fp_dist, box_q_dist = [], []
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
                'Var Name': var_name,
                'Var Type': 'Weight' if is_weight else 'Activation',
                'Corresponding Weight Name': self.acts_weight_map[var_name]
                if not is_weight else None,
                'FP32 Min': fp_min,
                'FP32 Max': fp_max,
                'FP32 Mean': fp_mean,
                'FP32 Std': fp_std,
                'Quantized Min': q_min,
                'Quantized Max': q_max,
                'Quantized Mean': q_mean,
                'Quantized Std': q_std,
                'Diff Min': diff_min,
                'Diff Max': diff_max,
                'Diff Mean': diff_mean,
                'Diff Std': diff_std,
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
            sample_num = len(box_fp_tensor) if len(
                box_fp_tensor) < 1000 else 1000
            box_fp_tensor = random.sample(list(box_fp_tensor), sample_num)
            box_q_tensor = random.sample(list(box_q_tensor), sample_num)
            box_fp_dist.append(box_fp_tensor)
            box_q_dist.append(box_q_tensor)

            # for histplot
            _, hist_edges = np.histogram(
                fp_tensor.copy(), bins=50, range=(fp_min, fp_max))
            hist_fp_dist[var_name] = [fp_tensor.flatten(), hist_edges]
            _, hist_edges = np.histogram(
                quant_tensor.copy(), bins=50, range=(q_min, q_max))
            hist_q_dist[quant_name] = [quant_tensor.flatten(), hist_edges]

        return statistic, box_fp_dist, box_q_dist, hist_fp_dist, hist_q_dist

    def plot_box_distribution(self, distribution, labels, save_name):
        all_values = sum(distribution, [])
        max_value = np.max(all_values)
        min_value = np.min(all_values)
        pdf_path = os.path.join(self.save_dir, save_name)
        with PdfPages(pdf_path) as pdf:
            for i in range(0, len(distribution), 20):
                r = i + 20 if i + 20 < len(distribution) else len(distribution)
                plt.boxplot(
                    distribution[i:r],
                    labels=labels[i:r],
                    showbox=True,
                    patch_artist=True)
                plt.xticks(rotation=90)
                plt.tick_params(axis='x')
                plt.ylim([min_value, max_value])
                if 'act' in save_name:
                    plt.xlabel('Activation Name')
                else:
                    plt.xlabel('Weight Name')
                plt.ylabel("Box Distribution")
                plt.tight_layout()
                plt.show()
                pdf.savefig()
                plt.close()
        _logger.info('Distribution plots is saved in {}'.format(pdf_path))

    def plot_hist_distribution(self, hist_data, save_name):
        pdf_path = os.path.join(self.save_dir, save_name)
        with PdfPages(pdf_path) as pdf:
            for name in hist_data:
                plt.hist(hist_data[name][0], bins=hist_data[name][1])
                plt.xlabel(name)
                plt.ylabel("Probability")
                locs, _ = plt.yticks()
                plt.yticks(locs, np.round(locs / len(hist_data[name][0]), 3))
                if 'act' in save_name:
                    plt.title("Hist of Activation {}".format(name))
                else:
                    plt.title("Hist of Weight {}".format(name))
                plt.show()
                pdf.savefig()
                plt.close()
        _logger.info('Histogram plot is saved in {}'.format(pdf_path))

    def get_target_quant_model(self, target_metric):
        _logger.info(
            'Start to Find quantized model that satisfies the target metric.')
        _logger.info(
            'Make sure that you are using full eval dataset to get target quantized model.'
        )
        skip_list = []
        if self.quant_layer_metrics:
            rank_list = sorted(
                self.quant_layer_metrics,
                key=self.quant_layer_metrics.get,
                reverse=True)
        else:
            _logger.info(
                'Analyse metric error before get target quantized model.')
            self.metric_error_analyse()

        while len(rank_list) > 0:
            skip_list.append(rank_list.pop(0))
            _logger.info('Skip Ops: {}'.format(skip_list))
            executor = paddle.static.Executor(self.places)
            post_training_quantization = self.create_ptq(executor, skip_list)
            program = post_training_quantization.quantize()

            _logger.info('Evaluating...')
            quant_metric = self.eval_function(executor, program, self.feed_list,
                                              self.fetch_list)
            _logger.info("Current eval metric: {}, the target metric: {}".
                         format(quant_metric, target_metric))
            if quant_metric >= target_metric:
                quantize_model_path = os.path.join(self.save_dir,
                                                   'target_quant_model')
                _logger.info(
                    'The quantized model satisfies the target metric and is saved to {}'.
                    format(quantize_model_path))
                post_training_quantization.save_quantized_model(
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

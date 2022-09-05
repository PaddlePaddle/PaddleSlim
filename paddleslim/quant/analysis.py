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
import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.framework import IrGraph
from paddle.fluid.executor import global_scope
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle.fluid.contrib.slim.quantization.utils import _get_op_input_var_names, load_variable_data
from .quanter import quant_post
from ..core import GraphWrapper
from ..common import get_logger
from ..common import get_feed_vars, wrap_dataloader, load_inference_model, get_model_dir

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["AnalysisQuant"]


class AnalysisQuant(object):
    def __init__(self,
                 model_dir,
                 model_filename=None,
                 params_filename=None,
                 eval_function=None,
                 data_loader=None,
                 save_dir='analysis_results',
                 checkpoint_name='analysis_checkpoint.pkl',
                 num_histogram_plots=10,
                 ptq_config=None):
        """
        AnalysisQuant provides to analysis the sensitivity of each op in the model.
        
        Args:
            model_dir(str): the path of fp32 model that will be quantized, it can also be '.onnx'
            model_filename(str, optional): the model file name of the fp32 model
            params_filename(str, optional): the parameter file name of the fp32 model
            eval_function(function): eval function, define by yourself to return the metric of the inference program, can be used to judge the metric of quantized model.  (TODO: optional)
            data_loader(Python Generator, Paddle.io.DataLoader, optional): the
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time
            save_dir(str, optional): the output dir that stores the analyzed information
            checkpoint_name(str, optional): the name of checkpoint file that saves analyzed information and avoids break off while ananlyzing
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
        self.checkpoint_name = os.path.join(save_dir, checkpoint_name)
        self.quant_layer_metrics = {}
        self.num_histogram_plots = num_histogram_plots
        self.ptq_config = ptq_config
        self.batch_nums = ptq_config[
            'batch_nums'] if 'batch_nums' in ptq_config else 10

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

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

        # evaluate before quant 
        # TODO: self.eval_function can be None
        if self.eval_function is not None:
            self.base_metric = self.eval_function(
                executor, program, self.feed_list, self.fetch_list)
            _logger.info('before quantized, the accuracy of the model is: {}'.
                         format(self.base_metric))

        # quant and evaluate after quant (skip_list = None)
        post_training_quantization = PostTrainingQuantization(
            executor=executor,
            data_loader=self.data_loader,
            model_dir=self.model_dir,
            model_filename=self.model_filename,
            params_filename=self.params_filename,
            skip_tensor_list=None,
            algo='avg',  #fastest
            **self.ptq_config)
        program = post_training_quantization.quantize()
        self.quant_metric = self.eval_function(executor, program,
                                               self.feed_list, self.fetch_list)
        _logger.info('after quantized, the accuracy of the model is: {}'.format(
            self.quant_metric))

        # get quantized weight and act var name
        self.quantized_weight_var_name = post_training_quantization._quantized_weight_var_name
        self.quantized_act_var_name = post_training_quantization._quantized_act_var_name
        executor.close()

        # load tobe_analyized_layer from checkpoint 
        self.load_checkpoint()
        self.tobe_analyized_layer = self.quantized_weight_var_name - set(
            list(self.quant_layer_metrics.keys()))
        self.tobe_analyized_layer = sorted(list(self.tobe_analyized_layer))

    def analysis(self):
        self.compute_quant_sensitivity()
        self.sensitivity_ranklist = sorted(
            self.quant_layer_metrics,
            key=self.quant_layer_metrics.get,
            reverse=False)

        _logger.info('Finished computing the sensitivity of the model.')
        for name in self.sensitivity_ranklist:
            _logger.info("quant layer name: {}, eval metric: {}".format(
                name, self.quant_layer_metrics[name]))

        analysis_file = os.path.join(self.save_dir, "analysis.txt")
        with open(analysis_file, "w") as analysis_ret_f:
            for name in self.sensitivity_ranklist:
                analysis_ret_f.write(
                    "quant layer name: {}, eval metric: {}\n".format(
                        name, self.quant_layer_metrics[name]))
        _logger.info('Analysis file is saved in {}'.format(analysis_file))
        self.calculate_histogram()

    def save_checkpoint(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.checkpoint_name, 'wb') as f:
            pickle.dump(self.quant_layer_metrics, f)
        _logger.info('save checkpoint to {}'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        with open(self.checkpoint_name, 'rb') as f:
            self.quant_layer_metrics = pickle.load(f)
        _logger.info('load checkpoint from {}'.format(self.checkpoint_name))
        return True

    def compute_quant_sensitivity(self):
        '''
        For each layer, quantize the weight op and evaluate the quantized model.
        '''
        for i, layer_name in enumerate(self.tobe_analyized_layer):
            _logger.info('checking {}/{} quant model: quant layer {}'.format(
                i + 1, len(self.tobe_analyized_layer), layer_name))
            skip_list = copy.copy(list(self.quantized_weight_var_name))
            skip_list.remove(layer_name)

            executor = paddle.static.Executor(self.places)
            post_training_quantization = PostTrainingQuantization(
                executor=executor,
                data_loader=self.data_loader,
                model_dir=self.model_dir,
                model_filename=self.model_filename,
                params_filename=self.params_filename,
                skip_tensor_list=skip_list,
                algo='avg',  #fastest
                **self.ptq_config)
            program = post_training_quantization.quantize()

            _logger.info('Evaluating...')
            quant_metric = self.eval_function(executor, program, self.feed_list,
                                              self.fetch_list)
            executor.close()
            _logger.info(
                "quant layer name: {}, eval metric: {}, the loss caused by this layer: {}".
                format(layer_name, quant_metric, self.base_metric -
                       quant_metric))
            self.quant_layer_metrics[layer_name] = quant_metric
            self.save_checkpoint()

    def get_act_name_by_weight(self, program, weight_names,
                               persistable_var_names):
        act_ops_names = []
        for op_name in weight_names:
            for block_id in range(len(program.blocks)):
                for op in program.blocks[block_id].ops:
                    var_name_list = _get_op_input_var_names(op)
                    if op_name in var_name_list:
                        for var_name in var_name_list:
                            if var_name not in persistable_var_names:
                                act_ops_names.append(var_name)
        return act_ops_names

    def get_hist_ops_name(self, graph, program):
        if self.num_histogram_plots <= 0:
            return []

        best_weight_ops = self.sensitivity_ranklist[::-1][:self.
                                                          num_histogram_plots]
        worst_weight_ops = self.sensitivity_ranklist[:self.num_histogram_plots]

        persistable_var_names = []
        for var in program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)

        best_act_ops = self.get_act_name_by_weight(program, best_weight_ops,
                                                   persistable_var_names)
        worst_act_ops = self.get_act_name_by_weight(program, worst_weight_ops,
                                                    persistable_var_names)
        return [best_weight_ops, best_act_ops, worst_weight_ops, worst_act_ops]

    def collect_ops_histogram(self, scope, ops):
        hist = {}
        for var_name in ops:
            var_tensor = load_variable_data(scope, var_name)
            var_tensor = np.array(var_tensor)
            min_v = float(np.min(var_tensor))
            max_v = float(np.max(var_tensor))
            var_tensor = var_tensor.flatten()
            _, hist_edges = np.histogram(
                var_tensor.copy(),
                bins=self.histogram_bins,
                range=(min_v, max_v))
            hist[var_name] = [var_tensor, hist_edges]
        return hist

    def calculate_histogram(self):
        '''
        Sample histograms for the weight and corresponding act tensors
        '''
        devices = paddle.device.get_device().split(':')[0]
        places = paddle.device._convert_to_place(devices)
        executor = paddle.static.Executor(places)

        [program, feed_list, fetch_list]= load_inference_model( \
            self.model_dir, \
            executor=executor, \
            model_filename=self.model_filename, \
            params_filename=self.params_filename)

        scope = global_scope()

        graph = IrGraph(core.Graph(program.desc), for_test=False)
        ops_tobe_draw_hist = self.get_hist_ops_name(graph, program)
        if not ops_tobe_draw_hist:
            return

        for var in program.list_vars():
            if var.name in self.quantized_act_var_name:
                var.persistable = True

        # sample before collect histogram
        batch_id = 0
        for data in self.data_loader():
            executor.run(program=program,
                         feed=data,
                         fetch_list=fetch_list,
                         return_numpy=False,
                         scope=scope)
            batch_id += 1
            if batch_id >= self.batch_nums:
                break

        pdf_names = [
            'best_weight_hist_result.pdf',
            'best_act_hist_result.pdf',
            'worst_weight_hist_result.pdf',
            'worst_act_hist_result.pdf',
        ]
        for ops, save_pdf_name in zip(ops_tobe_draw_hist, pdf_names):
            hist_data = self.collect_ops_histogram(scope, ops)
            self.draw_pdf(hist_data, save_pdf_name)

    def draw_pdf(self, hist_data, save_pdf_name):
        pdf_path = os.path.join(self.save_dir, save_pdf_name)
        with PdfPages(pdf_path) as pdf:
            for name in hist_data:
                plt.hist(hist_data[name][0], bins=hist_data[name][1])
                plt.xlabel(name)
                plt.ylabel("frequency")
                plt.title("Hist of variable {}".format(name))
                plt.show()
                pdf.savefig()
                plt.close()
        _logger.info('Histogram plot is saved in {}'.format(pdf_path))

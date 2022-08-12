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
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddleslim.quant.quanter import quant_post
from paddleslim.core import GraphWrapper
from paddle.fluid.contrib.slim.quantization.utils import _get_op_input_var_names, load_variable_data
from paddleslim.common import get_logger
from ..common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["AnalysisQuant"]


def _valid_format(data):
    is_dict = isinstance(data, dict)
    list_with_one_dict = isinstance(
        data, list) and len(data) == 1 and isinstance(data[0], dict)
    return is_dict or list_with_one_dict


def wrap_dataloader(dataloader, names):
    """Create a wrapper of dataloader if the data returned by the dataloader is not a dict.
    And the names will be the keys of dict returned by the wrapper.
    """
    if dataloader is None:
        return dataloader
    data = next(dataloader())
    if _valid_format(data):
        return dataloader

    if isinstance(data, Iterable):
        assert len(data) == len(
            names
        ), f"len(data) == len(names), but got len(data): {len(data)} and len(names): {len(names)}"
    else:
        assert len(
            names
        ) == 1, f"The length of name should 1 when data is not Iterable but got {len(names)}"

    def gen():
        for i, data in enumerate(dataloader()):
            if not isinstance(data, Iterable):
                data = [data]
            yield dict((name_, np.array(data_))
                       for name_, data_ in zip(names, data))

    return gen


class AnalysisQuant(object):
    def __init__(
            self,
            model_dir,
            model_filename='model.pdmodel',
            params_filename='model.pdiparams',
            eval_function=None,
            quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
            batch_size=10,
            batch_nums=10,
            data_loader=None,
            save_dir='results',
            checkpoint_name='analysis_checkpoint.pkl',
            num_histogram_plots=10, ):
        """
        AnalysisQuant provides to analysis the sensitivity of each op in the model.
        
        Args:
            model_dir(str): the path of fp32 model that will be quantized
            model_filename(str): the model file name of the fp32 model
            params_filename(str): the parameter file name of the fp32 model
            eval_function(function): eval function, define by yourself to return the metric of the inference program, can be used to judge the metric of quantized model.  (TODO: optional)
            quantizable_op_type(list, optional): op types that can be quantized
            batch_size(int, optional): the batch size of DataLoader, default is 10
            data_loader(Python Generator, Paddle.io.DataLoader, optional): the
                Generator or Dataloader provides calibrate data, and it could
                return a batch every time
            save_dir(str, optional): the output dir that stores the analyzed information
            checkpoint_name(str, optional): the name of checkpoint file that saves analyzed information and avoids break off while ananlyzing
            num_histogram_plots: the number histogram plots you want to visilize, the plots will show in one PDF file in the save_dir
        """
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.batch_nums = batch_nums
        self.quantizable_op_type = quantizable_op_type
        self.is_full_quantize = True
        self.histogram_bins = 1000
        self.save_dir = save_dir
        self.eval_function = eval_function
        self.quant_layer_names = []
        self.checkpoint_name = os.path.join(save_dir, checkpoint_name)
        self.quant_layer_metrics = {}
        self.batch_size = batch_size
        self.batch_nums = batch_nums
        self.num_histogram_plots = num_histogram_plots

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        devices = paddle.device.get_device().split(':')[0]
        self.places = paddle.device._convert_to_place(devices)
        executor = paddle.static.Executor(self.places)

        # load model 
        [program, self.feed_list, self.fetch_list]= paddle.fluid.io.load_inference_model( \
            dirname=model_dir, \
            executor=executor, \
            model_filename=model_filename, \
            params_filename=params_filename)

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
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo='avg',  # fastest
            quantizable_op_type=self.quantizable_op_type,
            weight_quantize_type='abs_max',
            skip_tensor_list=None)
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
        self.draw_pdf()

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
                batch_size=self.batch_size,
                batch_nums=self.batch_nums,
                algo='avg',  # fastest
                quantizable_op_type=self.quantizable_op_type,
                weight_quantize_type='abs_max',
                skip_tensor_list=skip_list)
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

    def get_sensitive_ops_name(self, graph, program):
        sensitive_weight_ops = self.sensitivity_ranklist[:self.
                                                         num_histogram_plots]
        sensitive_act_ops = []
        persistable_var_names = []
        persistable_var_names = []
        for var in program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)
        for op_name in sensitive_weight_ops:
            for block_id in range(len(program.blocks)):
                for op in program.blocks[block_id].ops:
                    var_name_list = _get_op_input_var_names(op)
                    if op_name in var_name_list:
                        for var_name in var_name_list:
                            if var_name not in persistable_var_names:
                                sensitive_act_ops.append(var_name)
        return sensitive_act_ops, sensitive_weight_ops

    def calculate_histogram(self):
        '''
        Sample histograms for the weight and corresponding act tensors
        '''
        devices = paddle.device.get_device().split(':')[0]
        places = paddle.device._convert_to_place(devices)
        executor = paddle.static.Executor(places)

        [program, feed_list, fetch_list]= paddle.fluid.io.load_inference_model( \
            dirname=self.model_dir, \
            executor=executor, \
            model_filename=self.model_filename, \
            params_filename=self.params_filename)

        scope = global_scope()

        graph = IrGraph(core.Graph(program.desc), for_test=False)
        self.sensitive_act_ops, self.sensitive_weight_ops = self.get_sensitive_ops_name(
            graph, program)

        for var in program.list_vars():
            if var.name in self.quantized_act_var_name:
                var.persistable = True

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

        self.weight_histogram = {}
        self.act_histogram = {}
        for var_name in self.sensitive_act_ops:
            var_tensor = load_variable_data(scope, var_name)
            var_tensor = np.array(var_tensor)
            min_v = float(np.min(var_tensor))
            max_v = float(np.max(var_tensor))
            var_tensor = var_tensor.flatten()
            _, hist_edges = np.histogram(
                var_tensor.copy(),
                bins=self.histogram_bins,
                range=(min_v, max_v))
            self.act_histogram[var_name] = [var_tensor, hist_edges]

        for var_name in self.sensitive_weight_ops:
            var_tensor = load_variable_data(scope, var_name)
            var_tensor = np.array(var_tensor)
            min_v = float(np.min(var_tensor))
            max_v = float(np.max(var_tensor))
            var_tensor = var_tensor.flatten()
            _, hist_edges = np.histogram(
                var_tensor.copy(),
                bins=self.histogram_bins,
                range=(min_v, max_v))
            self.weight_histogram[var_name] = [var_tensor, hist_edges]

    def draw_pdf(self):
        pdf_path_a = os.path.join(self.save_dir, 'act_hist_result.pdf')
        pdf_path_w = os.path.join(self.save_dir, 'weight_hist_result.pdf')
        with PdfPages(pdf_path_a) as pdf:
            for name in self.act_histogram:
                plt.hist(
                    self.act_histogram[name][0],
                    bins=self.act_histogram[name][1])
                plt.xlabel(name)
                plt.ylabel("frequency")
                plt.title("Hist of variable {}".format(name))
                plt.show()
                pdf.savefig()
            plt.close()
        with PdfPages(pdf_path_w) as pdf:
            for name in self.weight_histogram:
                plt.hist(
                    self.weight_histogram[name][0],
                    bins=self.weight_histogram[name][1])
                plt.xlabel(name)
                plt.ylabel("frequency")
                plt.title("Hist of variable {}".format(name))
                plt.show()
                pdf.savefig()
            plt.close()
        _logger.info('Histogram plots are saved in {} and {}'.format(
            pdf_path_a, pdf_path_w))

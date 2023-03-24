"""Define latency predictor that predict the latency of model on devices.
"""
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import time
import os
import pip
import platform
import logging
import pickle
import shutil
import subprocess
import warnings
import urllib.request as request
import ssl
import paddle
from .parse_ops import get_key_from_op
from .extract_features import get_data_from_tables, get_features_from_paramkey
from ._utils import opt_model, load_predictor, nearest_interpolate, _get_download
from ..common import get_logger
from ..core import GraphWrapper
__all__ = ["LatencyPredictor", "TableLatencyPredictor"]

_logger = get_logger(__name__, level=logging.INFO)

TABLE_URL = 'https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/'


def format_Warning(message, category, filename, lineno, line=''):
    return str(filename) + ':' + str(
        lineno) + ': ' + category.__name__ + ': ' + str(message) + '\n'


warnings.formatwarning = format_Warning


class LatencyPredictor(object):
    """Base class of latency predictor.
    """

    def predict(self, model):
        """Get latency of model. It is an abstract method.

        Args:
            model: The model to be evaluated.

        Returns:
            latency(float): The latency of given model on current evaluator.
        """
        raise NotImplementedError('Abstract method.')

    def _get_key_info_from_graph(self, graph):
        graph_keys = []
        for op in graph.ops():
            param_key = get_key_from_op(op)
            graph_keys.append(param_key)
        return graph_keys


class TableLatencyPredictor(LatencyPredictor):
    """The preditor used to get pbmodel's latency on some devices and infer engines.

    Args:
        table_file(str): The path of file that records the device latency of operators.
    """
    hardware_list = ['SD625', 'SD710', 'RK3288']

    def __init__(self, table_file='SD710'):
        self._check_opt_model()
        self.table_file = table_file
        self.table_dict = {}
        self.hardware = None
        self.threads = None
        self.predictor_state = False
        self.predictor = {}
        self._initial_table()

    @classmethod
    def add_hardware(cls, hardware):
        cls.hardware_list.append(hardware)

    def _check_opt_model(self):
        if platform.system().lower() == 'windows':
            raise NotImplementedError(
                'latency predictor does NOT support running on Windows.')
        elif platform.system().lower() == 'darwin':
            py_version = platform.python_version().split('.')
            if int(py_version[0]) != 3 or int(py_version[1]) != 9:
                raise NotImplementedError(
                    'Latency predictor does NOT support running on macOS when python version is not 3.9.'
                )

        _logger.info("pip install paddleslim-opt-tools")
        out = shutil.which('paddle_lite_opt')
        if out is None:
            from pip._internal import main
            main(['install', 'paddleslim-opt-tools'])

    def _initial_table(self):
        if self.table_file in TableLatencyPredictor.hardware_list:
            self.hardware = self.table_file
            self.threads = 4
            self.table_file = f'{self.hardware}_threads_4_power_mode_0.pkl'
            self.predictor_state = True
            url = TABLE_URL + self.table_file
            while not (os.path.exists(self.table_file)):
                if not _get_download(url, self.table_file):
                    time.sleep(1)
                    continue

            print('Successfully download {}!'.format(self.table_file))
        assert os.path.exists(
            self.table_file
        ), f'{self.table_file} does not exist. If you want to use our table files, please set \'table_file\' in {TableLatencyPredictor.hardware_list}'
        with open(self.table_file, 'rb') as f:
            self.table_dict = pickle.load(f)

        print('Successfully load {}'.format(self.table_file))

    def _change_table(self, threads=4):
        assert threads == 4, 'Only 4 threads are available now.'
        self.table_file = f'{self.hardware}_threads_{threads}_power_mode_0.pkl'
        if not os.path.exists(self.table_file):
            subprocess.call(
                f'wget https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/{self.table_file}',
                shell=True)

        with open(self.table_file, 'rb') as f:
            self.table_dict = pickle.load(f)

        print('Successfully loaded {}'.format(self.table_file))

    def _get_input_shape(self, graph):
        in_shape = []
        for op in graph.ops():
            param_key = get_key_from_op(op)
            if param_key != '':
                in_shape = op.all_inputs()[-1].shape()
                break
        return in_shape

    def _preload_predictor(self, data_type='fp32'):
        op_types = [
            'depthwise_conv2d', 'conv2d', 'pool2d', 'matmul', 'elementwise_add',
            'elementwise_mul', 'concat', 'calib', 'swish'
        ]
        op_dir = self.table_file.split('.')[0] + '_batchsize_1'
        for op_type in op_types:
            if data_type == 'fp32' and op_type == 'calib':
                continue
            model = load_predictor(op_type, op_dir, data_type)
            key = op_type
            if 'conv2d' in op_type:
                key = f'{op_type}_{data_type}'
            self.predictor[key] = model

    def predict(self,
                model_file,
                param_file,
                data_type,
                threads=4,
                input_shape=None):
        """predict the latency of the model
        
        Args:
            model_file(str), param_file(str): The inference model(*.pdmodel, *.pdiparams).
            data_type(str): Data type, fp32, fp16 or int8.
            threads(int): Threads num.
            input_shape(list): Generally, the input shape is confirmed when saving the inference model and the parameter is only effective for input shape that has variable length.
        Returns:
            latency(float): The latency of the model.
        """
        assert data_type in ['fp32', 'int8', 'fp16'
                             ], f'data_type must be one of [fp32, int8, fp16]'

        if self.hardware and self.threads != threads:
            self._change_table(threads)

        if self.predictor_state and f'conv2d_{data_type}' not in self.predictor:
            self._preload_predictor(data_type)

        enable_fp16 = True if data_type == 'fp16' else False
        pbmodel_file = opt_model(
            model_file=model_file,
            param_file=param_file,
            optimize_out_type='protobuf',
            enable_fp16=enable_fp16)

        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            _program = paddle.static.Program.parse_from_string(f.read())

        graph = GraphWrapper(_program)

        if input_shape != None:
            ori_shape = self._get_input_shape(graph)
            assert ori_shape == input_shape, "The parameter \'input_shape\' dosn't work for now. The input shape is fixed when saving the inference model"

        latency = 0.0
        new_op = {}
        for op in graph.ops():
            param_key = get_key_from_op(op)
            if param_key == '':
                continue
            if param_key == None:
                if op.type() in new_op:
                    new_op[op.type()] += 1
                else:
                    new_op.update({op.type(): 1})
                continue
            if param_key in self.table_dict:
                latency += self.table_dict[param_key]
            elif self.predictor_state:
                latency += self.op_predictor(op.type(), param_key, data_type)
        if len(new_op) != 0:
            warnings.warn(
                "These ops are not currently supported. Please raise an issue in PaddleSlim if you find the CalledTimes is large enough to affect the accuracy."
            )
            warnings.warn("OperatorType\tCalledTimes")
            for key in new_op:
                warnings.warn(f"{key.ljust(15)}\t{new_op[key]}")
        shutil.rmtree(os.path.dirname(pbmodel_file))
        return latency

    def op_predictor(self, op_type, param_key, data_type):
        """predict the latency of the operator which is not in the table
        
        Args:
            op_type: The operator's type
            param_key: The operator's parameter information.
            data_type: Data type, fp32 or int8.
        Returns:
            latency(float): The latency of the operator.
        """

        latency = 0.0
        if op_type in [
                'depthwise_conv2d', 'conv2d', 'pool2d', 'matmul',
                'elementwise_add', 'elementwise_mul', 'concat', 'calib', 'swish'
        ]:
            key = op_type
            if 'conv2d' in op_type:
                key = f'{op_type}_{data_type}'
            predictor = self.predictor[key]
            features = get_features_from_paramkey(param_key, op_type, data_type)
            latency = predictor.predict([features])
        else:
            data = get_data_from_tables(
                table_dict=self.table_dict,
                op_type=op_type,
                data_type=data_type)
            features = get_features_from_paramkey(param_key, op_type, data_type)
            latency = nearest_interpolate(features, data)
            if latency is None:
                return 0.

        return latency

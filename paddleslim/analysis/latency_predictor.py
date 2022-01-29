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

import os
import pickle
import time
import subprocess
from .parse_ops import get_key_from_op
from .extract_features import get_data_from_tables, get_features_from_paramkey
from ._utils import opt_model, load_predictor, nearest_interpolate
import paddle
import paddleslim
__all__ = ["LatencyPredictor", "TableLatencyPredictor"]


class LatencyPredictor(object):
    """Base class of latency predictor.
    """

    def predict_latency(self, model):
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
        table_file(str): The path of file that records the devices latency of operators.
    """

    def __init__(self, table_file='SD710'):
        self.table_file = table_file
        self.table_dict = {}
        self.hardware = None
        self.threads = None
        self._initial_table()
        self.predictor_state = False

    def _initial_table(self):
        if self.table_file in ['SD625', 'SD710', 'SD845', 'SD865']:
            self.hardware = self.table_file
            self.threads = 4
            self.table_file = f'{self.hardware}_threads_4_power_mode_0.pkl'
            if not os.path.exists(self.table_file):
                subprocess.call(
                    f'wget https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/{self.table_file}',
                    shell=True)

        assert os.path.exists(
            self.table_file
        ), f'{self.table_file} is not existed. If you want to use our table files, please set \'table_file\' in [SD625, SD710, SD845, SD865]'
        with open(self.table_file, 'rb') as f:
            self.table_dict = pickle.load(f)

        print('Successfully load {}'.format(self.table_file))

    def _change_table(self, threads=4):
        self.table_file = f'{self.hardware}_threads_{threads}_power_mode_0.pkl'
        if not os.path.exists(self.table_file):
            subprocess.call(
                f'wget https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/{self.table_file}',
                shell=True)

        with open(self.table_file, 'rb') as f:
            self.table_dict = pickle.load(f)

        print('Successfully load {}'.format(self.table_file))

    def _get_input_shape(self, graph):
        in_shape = []
        for op in graph.ops():
            param_key = get_key_from_op(op)
            if param_key != '':
                in_shape = op.all_inputs()[-1].shape()
                break
        return in_shape

    def set_predictor_state(self, state=False):
        """Adjust the op predictor‘s state. Default: False.
        """
        self.predictor_state = state

    def predict(self,
                model_file,
                param_file,
                data_type,
                threads=4,
                input_shape=[1, 3, 224, 224]):
        """predict the latency of the model
        
        Args:
            model_file(str), param_file(str): The inference model(*.pdmodel, *.pdiparams).
            data_type(str): Data type, fp32 or int8. Default : fp32
            threads(int): threads num
            input_shape(list): Generally, the input shape is confirmed when saving the inference model and the parameter is only effective for variable length input shape.
        Returns:
            latency(float): The latency of the model.
        """
        assert data_type in ['fp32', 'int8'
                             ], f'data_type must be one of [fp32, int8]'

        if self.hardware and self.threads != threads:
            self._change_table(threads)

        pbmodel_file = opt_model(
            model_file=model_file,
            param_file=param_file,
            optimize_out_type='protobuf', )

        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            fluid_program = paddle.fluid.framework.Program.parse_from_string(
                f.read())

        graph = paddleslim.core.GraphWrapper(fluid_program)

        if input_shape != [1, 3, 224, 224]:
            ori_shape = self._get_input_shape(graph)
            assert ori_shape == input_shape, "The parameter \'input_shape\' dosn't work now. The input shape is confirmed when saving the inference model"

        latency = 0.0
        for op in graph.ops():
            param_key = get_key_from_op(op)
            if param_key == '':
                continue
            if param_key in self.table_dict:
                latency += self.table_dict[param_key]
            elif self.predictor_state:
                latency += self.op_predictor(op.type(), param_key, data_type)
            else:
                raise AssertionError(f'{param_key} is not in the table.')

        return latency

    def op_predictor(self, op_type, param_key, data_type):
        """predict the latency of the operator which is not in the table
        
        Args:
            op_type: The operator's type
            param_key: The operator's parameter information.
            data_type: Data type, fp32 or int8. Default : int8
        Returns:
            latency(float): The latency of the operator.
        """

        latency = 0.0
        op_dir = self.table_file.split('.')[0] + '_batchsize_1'
        if op_type in [
                'depthwise_conv2d', 'conv2d', 'pool2d', 'matmul',
                'elementwise_add', 'elementwise_mul', 'concat', 'calib', 'swish'
        ]:
            predictor = load_predictor(op_type, op_dir, data_type)
            features = get_features_from_paramkey(param_key, op_type, data_type)
            latency = predictor.predict([features])
        else:
            data = get_data_from_tables(
                table_dict=self.table_dict,
                op_type=op_type,
                data_type=data_type)
            features = get_features_from_paramkey(param_key, op_type, data_type)
            latency = nearest_interpolate(features, data)
            assert latency != None, f'{param_key} is not in the table.'

        return latency

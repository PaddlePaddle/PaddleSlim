"""Define latency predictor that predict the latency of model on devices.
"""
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from _utils import get_key_from_op, save_cls_model, save_det_model, save_seg_model
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
            param_key = get_key_from_op(args, op)
            graph_keys.append(graph_keys)
        return graph_keys


class TableLatencyPredictor(LatencyPredictor):
    """The preditor used to get pbmodel's latency on some devices and infer engines.

    Args:
      table_file(str): The path of file that records the devices latency of operators.
      opt_path(str): The path of opt tool to convert a paddle model to an optimized pbmodel that fuses operators.
    """

    def __init__(self, table_file, opt_path):
        self.table_file = table_file
        self.opt_path = opt_path
        self.table_dict = {}
        self._read_table()
        self.det_multi_input = False
    
    def _read_table(self):
        if os.path.exists(self.table_file):
            with open(self.table_file, 'rb') as f:
                self.table_dict = pickle.load(f)
            print('Successfully load {}'.format(self.table_file))
        else:
            assert os.path.exists(self.table_file), f'{self.table_file} is not existed.'
        
    def set_det_multi_input(self, det_multi_input):
        """If a detection model has multiple input, the det_multi_input should be True.
           Since these models can only be tested by detect_system tool instead of paddlelite's benchmark tool
        """
        self.det_multi_input = det_multi_input

    def opt_model(self, model, input_shape, save_dir, data_type, task_type):
        """Convert the model graph to an optimized pbmodel by using opt tool.
        
        Args:
            model: The input model graph.
            input_shape(list): The input shape of model.
            save_dir: Where to save the pbmodel.
            data_type: Data type, fp32 or int8.
        Returns:
            pbmodel_file: The path of optimized pbmodel.
        """

        if task_type == 'cls':
            model_file, param_file = save_cls_model(model=model, input_shape=input_shape, save_dir=save_dir, data_type=data_type)

        elif task_type == 'det':
            model_file, param_file = save_det_model(model=model, input_shape=input_shape, save_dir=save_dir, data_type=data_type, det_multi_input=self.det_multi_input)

        elif task_type == 'seg':
            model_file, param_file = save_seg_model(model=model, input_shape=input_shape, save_dir=save_dir, data_type=data_type)

        else:
            assert task_type in ['cls', 'det', 'seg'], f'task_type must be one of [cls, det, seg]'

        pb_model = os.path.join(save_dir, f'{data_type}pbmodel')
        if not os.path.exists(pb_model):
            os.makedirs(pb_model)

        cmd = f'{self.opt_path} --model_file={model_file} --param_file={param_file}  --optimize_out_type=protobuf --optimize_out={pb_model} --valid_targets=arm'
        print(f'commands:{cmd}')
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = m.communicate()
        print(out, 'opt done!')

        pbmodel_file = os.path.join(pb_model, 'model')
        
        return pbmodel_file

    def predict_latency(self, model, input_shape=[1,3,224,224], save_dir='', data_type='int8', task_type='cls'):
        """predict the latency of the model
        
        Args:
            model: The input model graph.
            input_shape(list): The input shape of model. Default: [1,3,224,224].
            save_dir: Where to save the pbmodel.
            data_type: Data type, fp32 or int8. Default : int8
        Returns:
            latency(ms): The latency of the pbmodel.
        """

        assert data_type in ['fp32', 'int8'], f'data_type must be one of [fp32, int8]'
        assert task_type in ['cls', 'det', 'seg'], f'task_type must be one of [cls, det, seg]'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pbmodel_file = self.opt_model(model=model, input_shape=input_shape, save_dir=save_dir, data_type=data_type, task_type=task_type)

        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            program_desc_str = f.read()
            program = paddle.fluid.proto.framework_pb2.ProgramDesc.FromString(program_desc_str)
            fluid_program = paddle.fluid.framework.Program.parse_from_string(program_desc_str)

        graph = paddleslim.core.GraphWrapper(fluid_program)
        latency = 0.0
        for op in graph.ops(): 
            param_key = get_key_from_op(op)
            if param_key != '':
                assert param_key in self.table_dict, f'{param_key} is not in the tabel.'
                latency += self.table_dict[param_key]

        return latency

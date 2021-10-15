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
import sys, os
sys.path.append("../")
import unittest
import paddle
import paddleslim
from paddleslim.analysis import LatencyPredictor, TableLatencyPredictor
from paddle.vision.models import mobilenet_v1, mobilenet_v2

import subprocess

opt_tool = 'opt_ubuntu'  # use in linux
# opt_tool = 'opt_M1_mac'     # use in mac with M1 chip
# opt_tool = 'opt_intel_mac'  # use in mac with intel chip

if not os.path.exists(opt_tool):
    subprocess.call(
        f'wget https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/{opt_tool}',
        shell=True)
    subprocess.call(f'chmod +x {opt_tool}', shell=True)


class TestCase1(unittest.TestCase):
    def test_case1(self):
        paddle.disable_static()
        model = mobilenet_v1()
        predictor = TableLatencyPredictor(
            f'./{opt_tool}',
            hardware='845',
            threads=4,
            power_mode=3,
            batchsize=1)
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='fp32',
            task_type='cls')
        print(latency)
        assert latency == 41.92806607483133
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='int8',
            task_type='cls')
        assert latency == 36.65014722993898


class TestCase2(unittest.TestCase):
    def test_case2(self):
        paddle.disable_static()
        model = mobilenet_v2()
        predictor = TableLatencyPredictor(
            f'./{opt_tool}',
            hardware='845',
            threads=4,
            power_mode=3,
            batchsize=1)
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='fp32',
            task_type='cls')
        assert latency == 27.847896889217566
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='int8',
            task_type='cls')
        assert latency == 23.9698003601388


class TestCase3(unittest.TestCase):
    def test_case3(self):
        paddle.disable_static()
        model = mobilenet_v2()
        predictor = TableLatencyPredictor(
            f'./{opt_tool}',
            hardware='845',
            threads=4,
            power_mode=3,
            batchsize=1)
        pred = LatencyPredictor()
        pbmodel_file = predictor.opt_model(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='fp32',
            task_type='cls')
        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            program_desc_str = f.read()
            program = paddle.fluid.proto.framework_pb2.ProgramDesc.FromString(
                program_desc_str)
            fluid_program = paddle.fluid.framework.Program.parse_from_string(
                program_desc_str)
            graph = paddleslim.core.GraphWrapper(fluid_program)
            graph_keys = pred._get_key_info_from_graph(graph=graph)
            assert len(graph_keys) > 0


if __name__ == '__main__':
    unittest.main()

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
from paddle.nn import Conv2D
import subprocess

opt_tool = 'opt_ubuntu'  # use in linux
# opt_tool = 'opt_M1_mac'     # use in mac with M1 chip
# opt_tool = 'opt_intel_mac'  # use in mac with intel chip

if not os.path.exists(opt_tool):
    subprocess.call(
        f'wget https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/{opt_tool}',
        shell=True)
    subprocess.call(f'chmod +x {opt_tool}', shell=True)


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    x = paddle.reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])

    x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])

    x = paddle.reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


class ModelCase1(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase1, self).__init__()
        self.conv1 = Conv2D(58, 58, 1)
        self.conv2 = Conv2D(58, 58, 1)

    def forward(self, inputs):
        x1, x2 = paddle.split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        out = paddle.concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class ModelCase2(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase2, self).__init__()
        self.conv1 = Conv2D(3, 24, 3, stride=2, padding=1)

    def forward(self, inputs):
        inputs = inputs['image']
        return self.conv1(inputs)


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
        assert latency > 0
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='int8',
            task_type='cls')
        assert latency > 0


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
        assert latency > 0
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir='./model',
            data_type='int8',
            task_type='cls')
        assert latency > 0


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
            fluid_program = paddle.fluid.framework.Program.parse_from_string(
                program_desc_str)
            graph = paddleslim.core.GraphWrapper(fluid_program)
            graph_keys = pred._get_key_info_from_graph(graph=graph)
            assert len(graph_keys) > 0


class TestCase4(unittest.TestCase):
    def test_case4(self):
        paddle.disable_static()
        model = ModelCase1()
        predictor = TableLatencyPredictor(
            f'./{opt_tool}',
            hardware='845',
            threads=4,
            power_mode=3,
            batchsize=1)
        latency = predictor.predict_latency(
            model,
            input_shape=[1, 116, 28, 28],
            save_dir='./model',
            data_type='fp32',
            task_type='cls')
        assert latency > 0


class TestCase5(unittest.TestCase):
    def test_case5(self):
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
            task_type='seg')
        assert latency > 0


class TestCase6(unittest.TestCase):
    def test_case6(self):
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
            data_type='int8',
            task_type='seg')
        assert latency > 0


class TestCase7(unittest.TestCase):
    def test_case7(self):
        paddle.disable_static()
        model = ModelCase2()
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
            task_type='det')
        assert latency > 0


if __name__ == '__main__':
    unittest.main()

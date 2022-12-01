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
import sys
sys.path.append("../")
import os
import sys
import unittest
import paddle
from static_case import StaticCase
from paddleslim.nas import SANAS
from paddleslim.analysis import flops
import numpy as np


class TestDartsSpace(StaticCase):
    def setUp(self):
        paddle.enable_static()
        self.init_test_case()
        port = np.random.randint(8337, 8773)
        self.sanas = SANAS(
            configs=self.configs, server_addr=("", port), save_checkpoint=None)

    def init_test_case(self):
        self.configs = [('DartsSpace')]

    def test_search_space(self):
        ### unittest for next_archs
        next_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        token2arch_program = paddle.static.Program()

        with paddle.static.program_guard(next_program, startup_program):
            inputs = paddle.static.data(
                name='input', shape=[None, 3, 32, 32], dtype='float32')
            drop_path_prob = paddle.static.data(
                name="drop_path_prob", shape=[None, 1], dtype="float32")
            drop_path_mask = paddle.static.data(
                name="drop_path_mask", shape=[None, 20, 4, 2], dtype="float32")
            archs = self.sanas.next_archs()
            for arch in archs:
                output = arch(inputs, drop_path_prob, drop_path_mask, True, 10)
                inputs = output


class TestSearchSpace(StaticCase):
    def __init__(self, methodNmae="check", search_sapce_name=None):
        super(TestSearchSpace, self).__init__(methodNmae)
        paddle.enable_static()
        self.configs = [(search_sapce_name, {
            "input_size": 32,
            "output_size": 8,
            "block_num": 2
        })]
        port = np.random.randint(8337, 8773)
        self.sanas = SANAS(
            configs=self.configs, server_addr=("", port), save_checkpoint=None)

    def check(self):
        next_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(next_program, startup_program):
            inputs = paddle.static.data(
                name='input', shape=[None, 3, 32, 32], dtype='float32')
            archs = self.sanas.next_archs()
            for arch in archs:
                output = arch(inputs)
                inputs = output


search_space_suite = unittest.TestSuite()
search_space_suite.addTest(TestDartsSpace("test_search_space"))
search_space_suite.addTest(
    TestSearchSpace(search_sapce_name="InceptionABlockSpace"))
search_space_suite.addTest(
    TestSearchSpace(search_sapce_name="MobileNetV1Space"))
search_space_suite.addTest(
    TestSearchSpace(search_sapce_name="MobileNetV2Space"))
search_space_suite.addTest(TestSearchSpace(search_sapce_name="ResNetSpace"))
search_space_suite.addTest(
    TestSearchSpace(search_sapce_name="ResNetBlockSpace"))

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(search_space_suite)

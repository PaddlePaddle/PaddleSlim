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
import sys
import unittest
import paddle.fluid as fluid
from paddleslim.nas import SANAS
from paddleslim.analysis import flops
import numpy as np

def compute_op_num(program):
    params = {}
    for block in program.blocks:
        for param in block.all_parameters():
            if len(param.shape) == 4: 
                print(param.name, param.shape)
                params[param.name] = param.shape
    return params

class TestSANAS(unittest.TestCase):
    def setUp(self):
        self.init_test_case()
        port = np.random.randint(8337, 8773)
        self.sanas = SANAS(configs=self.configs, server_addr=("", port), save_checkpoint=None)

    def init_test_case(self):
        self.configs=[('MobileNetV2BlockSpace', {'block_mask':[0]})]
        self.filter_num = np.array([
            3, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 160, 192, 224,
            256, 320, 384, 512
        ])
        self.k_size = np.array([3, 5])
        self.multiply = np.array([1, 2, 3, 4, 5, 6])
        self.repeat = np.array([1, 2, 3, 4, 5, 6])

    def test_all_function(self):
        ### unittest for next_archs
        next_program = fluid.Program()
        startup_program = fluid.Program()
        token2arch_program = fluid.Program()

        with fluid.program_guard(next_program, startup_program):
            inputs = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
            archs = self.sanas.next_archs()
            for arch in archs:
                output = arch(inputs)
                inputs = output
        current_tokens = self.sanas.current_info()['current_tokens']
        print("current_token", current_tokens)

        conv_list = compute_op_num(next_program)
        print(len(conv_list))
        ### assert conv number
        print(current_tokens[2])
        print(self.repeat[current_tokens[2]])
        self.assertTrue((self.repeat[current_tokens[2]] * 3) ==  len(conv_list), "the number of conv is NOT match, the number compute from token: {}, actual conv number: {}".format(self.repeat[current_tokens[2]] * 3, len(conv_list)))

        ### unittest for reward
        self.assertTrue(self.sanas.reward(float(1.0)), "reward is False")

        ### uniitest for tokens2arch
        arch = self.sanas.tokens2arch(self.sanas.current_info()['current_tokens'])

        ### unittest for current_info
        current_info = self.sanas.current_info()
        self.assertTrue(isinstance(current_info, dict), "the type of current info must be dict, but now is {}".format(type(current_info)))

if __name__ == '__main__':
    unittest.main()

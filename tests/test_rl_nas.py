# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import paddle
import paddle.fluid as fluid
from paddleslim.nas import RLNAS
from paddleslim.analysis import flops
from static_case import StaticCase
import numpy as np


def compute_op_num(program):
    params = {}
    ch_list = []
    for block in program.blocks:
        for param in block.all_parameters():
            if len(param.shape) == 4:
                params[param.name] = param.shape
                ch_list.append(int(param.shape[0]))
    return params, ch_list


class TestRLNAS(StaticCase):
    def setUp(self):
        paddle.enable_static()
        self.init_test_case()
        port = np.random.randint(8337, 8773)
        self.rlnas = RLNAS(
            key='lstm',
            configs=self.configs,
            server_addr=("", port),
            is_sync=False,
            controller_batch_size=1,
            lstm_num_layers=1,
            hidden_size=10,
            temperature=1.0,
            save_controller=False)

    def init_test_case(self):
        self.configs = [('MobileNetV2BlockSpace', {'block_mask': [0]})]
        self.filter_num = np.array([
            3, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 160, 192, 224,
            256, 320, 384, 512
        ])
        self.k_size = np.array([3, 5])
        self.multiply = np.array([1, 2, 3, 4, 5, 6])
        self.repeat = np.array([1, 2, 3, 4, 5, 6])

    def check_chnum_convnum(self, program, current_tokens):
        channel_exp = self.multiply[current_tokens[0]]
        filter_num = self.filter_num[current_tokens[1]]
        repeat_num = self.repeat[current_tokens[2]]

        conv_list, ch_pro = compute_op_num(program)
        ### assert conv number
        self.assertTrue((repeat_num * 3) == len(
            conv_list
        ), "the number of conv is NOT match, the number compute from token: {}, actual conv number: {}".
                        format(repeat_num * 3, len(conv_list)))

        ### assert number of channels
        ch_token = []
        init_ch_num = 32
        for i in range(repeat_num):
            ch_token.append(init_ch_num * channel_exp)
            ch_token.append(init_ch_num * channel_exp)
            ch_token.append(filter_num)
            init_ch_num = filter_num

        self.assertTrue(
            str(ch_token) == str(ch_pro),
            "channel num is WRONG, channel num from token is {}, channel num come fom program is {}".
            format(str(ch_token), str(ch_pro)))

    def test_all_function(self):
        ### unittest for next_archs
        next_program = fluid.Program()
        startup_program = fluid.Program()
        token2arch_program = fluid.Program()

        with fluid.program_guard(next_program, startup_program):
            inputs = fluid.data(
                name='input', shape=[None, 3, 32, 32], dtype='float32')
            archs = self.rlnas.next_archs(1)[0]
            current_tokens = self.rlnas.tokens
            for arch in archs:
                output = arch(inputs)
                inputs = output
        self.check_chnum_convnum(next_program, current_tokens[0])

        ### unittest for reward
        self.assertTrue(self.rlnas.reward(float(1.0)), "reward is False")

        ### uniitest for tokens2arch
        with fluid.program_guard(token2arch_program, startup_program):
            inputs = fluid.data(
                name='input', shape=[None, 3, 32, 32], dtype='float32')
            arch = self.rlnas.tokens2arch(self.rlnas.tokens[0])
            for arch in archs:
                output = arch(inputs)
                inputs = output
        self.check_chnum_convnum(token2arch_program, self.rlnas.tokens[0])

    def test_final_archs(self):
        ### unittest for final_archs
        final_program = fluid.Program()
        final_startup_program = fluid.Program()
        with fluid.program_guard(final_program, final_startup_program):
            inputs = fluid.data(
                name='input', shape=[None, 3, 32, 32], dtype='float32')
            archs = self.rlnas.final_archs(1)[0]
            current_tokens = self.rlnas.tokens
            for arch in archs:
                output = arch(inputs)
                inputs = output
        self.check_chnum_convnum(final_program, current_tokens[0])


if __name__ == '__main__':
    unittest.main()

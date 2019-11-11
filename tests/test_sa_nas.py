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
import unittest
import paddle.fluid as fluid
from paddleslim.nas import SANAS
from paddleslim.nas import SearchSpaceFactory
from paddleslim.analysis import flops


class TestSANAS(unittest.TestCase):
    def test_nas(self):

        factory = SearchSpaceFactory()
        config0 = {'input_size': 224, 'output_size': 7, 'block_num': 5}
        config1 = {'input_size': 7, 'output_size': 1, 'block_num': 2}
        configs = [('MobileNetV2Space', config0), ('ResNetSpace', config1)]

        space = factory.get_search_space([('MobileNetV2Space', config0)])
        origin_arch = space.token2arch()[0]

        main_program = fluid.Program()
        s_program = fluid.Program()
        with fluid.program_guard(main_program, s_program):
            input = fluid.data(
                name="input", shape=[None, 3, 224, 224], dtype="float32")
            origin_arch(input)
        base_flops = flops(main_program)

        search_steps = 3
        sa_nas = SANAS(
            configs, max_flops=base_flops, search_steps=search_steps)

        for i in range(search_steps):
            archs = sa_nas.next_archs()
            main_program = fluid.Program()
            s_program = fluid.Program()
            with fluid.program_guard(main_program, s_program):
                input = fluid.data(
                    name="input", shape=[None, 3, 224, 224], dtype="float32")
                archs[0](input)
            sa_nas.reward(1)
            self.assertTrue(flops(main_program) < base_flops)


if __name__ == '__main__':
    unittest.main()

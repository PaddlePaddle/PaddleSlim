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
import sys
sys.path.append("../")
import unittest
from static_case import StaticCase
import paddle.fluid as fluid
from paddleslim import flops
import paddleslim.models as models


class TestModel(StaticCase):
    def __init__(self, model_name, flops, prefix=None,
                 method_name="test_model"):
        super(TestModel, self).__init__(method_name)
        self.model_name = model_name
        self.flops = flops
        self.prefix = prefix

    def test_model(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            if self.prefix is not None:
                model = models.__dict__[self.model_name](
                    prefix_name=self.prefix)
            else:
                model = models.__dict__[self.model_name]()
            model.net(input)
        print(flops(main_program))
        self.assertTrue(self.flops == flops(main_program))


suite = unittest.TestSuite()
suite.addTest(TestModel("ResNet34", 29097984, prefix=""))
suite.addTest(TestModel("ResNet34", 29097984, prefix="model1"))
suite.addTest(TestModel("MobileNet", 5110528))

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

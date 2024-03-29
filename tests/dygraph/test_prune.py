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
sys.path.append("../../")
import unittest
import paddle
from paddleslim import L1NormFilterPruner
from paddleslim.prune import Pruner


class TestPrune(unittest.TestCase):
    def __init__(self, methodName='runTest', ratios=None, net=None):
        super(TestPrune, self).__init__(methodName)
        self._net = net
        self._ratios = ratios

    def runTest(self):
        static_shapes = self.static_prune(self._net, self._ratios)
        dygraph_shapes = self.dygraph_prune(self._net, self._ratios)
        lazy_dygraph_shapes = self.dygraph_lazy_prune(self._net, self._ratios)
        all_right = True
        for _name, _shape in static_shapes.items():
            if dygraph_shapes[_name] != list(_shape):
                print(
                    f"name: {_name}; static shape: {_shape}, dygraph shape: {dygraph_shapes[_name]}"
                )
                all_right = False
        self.assertTrue(all_right)

    def dygraph_lazy_prune(self, net, ratios):
        if paddle.framework.core.is_compiled_with_cuda():
            paddle.set_device('gpu')
            self.dygraph_prune(net, ratios, apply="lazy")
        paddle.set_device('cpu')
        return self.dygraph_prune(net, ratios, apply="lazy")

    def dygraph_prune(self, net, ratios, apply="impretive"):
        paddle.disable_static()
        model = net(pretrained=False)
        pruner = L1NormFilterPruner(model, [1, 3, 16, 16])
        pruner.prune_vars(ratios, 0, apply=apply)
        shapes = {}
        for param in model.parameters():
            shapes[param.name] = param.shape
        pruner.restore()
        return shapes

    def static_prune(self, net, ratios):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program, startup_program):
                input = paddle.static.data(
                    name="image", shape=[None, 3, 16, 16])
                model = net(pretrained=False)
                out = model(input)

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        exe.run(startup_program, scope=scope)
        pruner = Pruner()
        main_program, _, _ = pruner.prune(
            main_program,
            scope,
            params=ratios.keys(),
            ratios=ratios.values(),
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None)

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape
        return shapes


def add_cases(suite):
    suite.addTest(
        TestPrune(
            net=paddle.vision.models.mobilenet_v1,
            ratios={"conv2d_22.w_0": 0.5,
                    "conv2d_8.w_0": 0.6}))
    suite.addTest(
        TestPrune(
            net=paddle.vision.models.resnet50,
            ratios={"conv2d_22.w_0": 0.5,
                    "conv2d_8.w_0": 0.6}))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()

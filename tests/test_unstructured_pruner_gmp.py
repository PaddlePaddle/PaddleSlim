import sys
sys.path.append("../")
import unittest
from static_case import StaticCase
import paddle.fluid as fluid
import paddle
from paddleslim.prune import UnstructuredPruner, GMPUnstructuredPruner
from layers import conv_bn_layer
import numpy as np


class TestUnstructuredPruner(StaticCase):
    def __init__(self, *args, **kwargs):
        super(TestUnstructuredPruner, self).__init__(*args, **kwargs)
        paddle.enable_static()
        self._gen_model()

    def _gen_model(self):
        self.main_program = paddle.static.default_main_program()
        self.startup_program = paddle.static.default_startup_program()
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            input = paddle.static.data(name='image', shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")

            conv7 = paddle.static.nn.conv2d_transpose(
                input=conv6, num_filters=16, filter_size=2, stride=2)

        place = paddle.static.cpu_places()[0]
        exe = paddle.static.Executor(place)
        self.scope = paddle.static.global_scope()
        exe.run(self.startup_program, scope=self.scope)

        configs = {
            'stable_iterations': 0,
            'pruning_iterations': 1000,
            'tunning_iterations': 1000,
            'resume_iteration': 500,
            'pruning_steps': 20,
            'initial_ratio': 0.05,
        }
        self.pruner = GMPUnstructuredPruner(
            self.main_program,
            scope=self.scope,
            place=place,
            configs=configs,
            ratio=0.55)
        print(self.pruner.ratio)
        self.assertGreater(self.pruner.ratio, 0.3)

    def test_unstructured_prune_gmp(self):
        last_ratio = 0.0
        ratio = 0.0
        while len(self.pruner.ratios_stack) > 0:
            self.pruner.step()
            last_ratio = ratio
            ratio = self.pruner.ratio
            self.assertGreaterEqual(ratio, last_ratio)
        self.assertEqual(ratio, 0.55)


if __name__ == '__main__':
    unittest.main()

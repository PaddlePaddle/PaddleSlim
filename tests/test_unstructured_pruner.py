import sys
sys.path.append("../")
import unittest
from static_case import StaticCase
import paddle.fluid as fluid
import paddle
from paddleslim.prune import UnstructuredPruner
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

            conv7 = fluid.layers.conv2d_transpose(
                input=conv6, num_filters=16, filter_size=2, stride=2)

        place = paddle.static.cpu_places()[0]
        exe = paddle.static.Executor(place)
        self.scope = paddle.static.global_scope()
        exe.run(self.startup_program, scope=self.scope)

        self.pruner = UnstructuredPruner(
            self.main_program, 'ratio', scope=self.scope, place=place)

    def test_unstructured_prune(self):
        for param in self.main_program.global_block().all_parameters():
            mask_name = param.name + "_mask"
            mask_shape = self.scope.find_var(mask_name).get_tensor().shape()
            self.assertTrue(tuple(mask_shape) == param.shape)

    def test_sparsity(self):
        ori_density = UnstructuredPruner.total_sparse(self.main_program)
        self.pruner.step()
        cur_density = UnstructuredPruner.total_sparse(self.main_program)
        cur_layer_density = self.pruner.sparse_by_layer(self.main_program)
        print('original density: {}.'.format(ori_density))
        print('current density: {}.'.format(cur_density))
        total = 0
        non_zeros = 0
        for param in self.main_program.all_parameters():
            total += np.product(param.shape)
            non_zeros += np.count_nonzero(
                np.array(self.scope.find_var(param.name).get_tensor()))
        self.assertEqual(cur_density, non_zeros / total)
        self.assertLessEqual(cur_density, ori_density)

        self.pruner.update_params()
        self.assertEqual(cur_density,
                         UnstructuredPruner.total_sparse(self.main_program))

    def test_summarize_weights(self):
        max_value = -float("inf")
        threshold = self.pruner.summarize_weights(self.main_program, 1.0)
        for param in self.main_program.global_block().all_parameters():
            max_value = max(
                max_value,
                np.max(np.array(self.scope.find_var(param.name).get_tensor())))
        print("The returned threshold is {}.".format(threshold))
        print("The max_value is {}.".format(max_value))
        self.assertEqual(max_value, threshold)


if __name__ == '__main__':
    unittest.main()

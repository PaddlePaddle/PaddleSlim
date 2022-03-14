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
            conv3 = conv_bn_layer(sum1, 8, 1, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 1, "conv4")
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
            self.main_program,
            'ratio',
            scope=self.scope,
            place=place,
            local_sparsity=True)
        self.pruner_conv1x1 = UnstructuredPruner(
            self.main_program,
            'ratio',
            scope=self.scope,
            place=place,
            prune_params_type='conv1x1_only',
            local_sparsity=False)
        self.pruner_mxn = UnstructuredPruner(
            self.main_program,
            'ratio',
            scope=self.scope,
            place=place,
            sparse_block=[2, 1],
            prune_params_type='conv1x1_only',
            local_sparsity=True)

    def test_unstructured_prune(self):
        for param in self.main_program.global_block().all_parameters():
            mask_name = param.name + "_mask"
            mask_shape = self.scope.find_var(mask_name).get_tensor().shape()
            self.assertTrue(tuple(mask_shape) == param.shape)

    def test_sparsity(self):
        ori_sparsity = UnstructuredPruner.total_sparse(self.main_program)
        self.pruner.step()
        self.pruner.update_params()
        cur_sparsity = UnstructuredPruner.total_sparse(self.main_program)
        cur_layer_sparsity = self.pruner.sparse_by_layer(self.main_program)
        print('original sparsity: {}.'.format(ori_sparsity))
        print('current sparsity: {}.'.format(cur_sparsity))
        total = 0
        non_zeros = 0
        for param in self.main_program.all_parameters():
            total += np.product(param.shape)
            non_zeros += np.count_nonzero(
                np.array(self.scope.find_var(param.name).get_tensor()))
        self.assertEqual(cur_sparsity, 1 - non_zeros / total)
        self.assertGreater(cur_sparsity, ori_sparsity)

        self.pruner.update_params()
        self.assertEqual(cur_sparsity,
                         UnstructuredPruner.total_sparse(self.main_program))

    def test_summarize_weights(self):
        max_value = -float("inf")
        threshold = self.pruner.summarize_weights(self.main_program, 1.0)
        for param in self.main_program.global_block().all_parameters():
            max_value = max(
                max_value,
                np.max(
                    np.abs(
                        np.array(self.scope.find_var(param.name).get_tensor(
                        )))))
        print("The returned threshold is {}.".format(threshold))
        print("The max_value is {}.".format(max_value))
        self.assertEqual(max_value, threshold)

    def test_unstructured_prune_conv1x1(self):
        print(self.pruner.skip_params)
        print(self.pruner_conv1x1.skip_params)
        self.assertTrue(
            self.pruner.skip_params < self.pruner_conv1x1.skip_params)

    def test_block_pruner_mxn(self):
        ori_sparsity = UnstructuredPruner.total_sparse_conv1x1(
            self.main_program)
        self.pruner_mxn.ratio = 0.50
        self.pruner_mxn.step()
        self.pruner_mxn.update_params()
        cur_sparsity = UnstructuredPruner.total_sparse_conv1x1(
            self.main_program)
        print('original sparsity: {}.'.format(ori_sparsity))
        print('current sparsity: {}.'.format(cur_sparsity))
        self.assertGreater(cur_sparsity, ori_sparsity)

    def test_sparsity_conv1x1(self):
        ori_sparsity = UnstructuredPruner.total_sparse_conv1x1(
            self.main_program)
        self.pruner.ratio = 0.99
        self.pruner.step()
        self.pruner.update_params()
        cur_sparsity = UnstructuredPruner.total_sparse_conv1x1(
            self.main_program)
        print('original sparsity: {}.'.format(ori_sparsity))
        print('current sparsity: {}.'.format(cur_sparsity))
        self.assertGreater(cur_sparsity, ori_sparsity)


if __name__ == '__main__':
    unittest.main()

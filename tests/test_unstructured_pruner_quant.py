import sys
sys.path.append("../")
import unittest
from static_case import StaticCase
import paddle.fluid as fluid
import paddle
from paddleslim.prune import UnstructuredPruner
from layers import conv_bn_layer
import numpy as np


class TestStaticMasks(StaticCase):
    def _update_masks(self, pruner, t):
        for param in pruner.masks:
            mask_name = pruner.masks[param]
            t_param = pruner.scope.find_var(param).get_tensor()
            t_mask = pruner.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            v_mask = (np.abs(v_param) < t).astype(v_param.dtype)
            t_mask.set(v_mask, pruner.place)

    def test_set_static_masks(self):
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            input = paddle.static.data(name='image', shape=[None, 3, 16, 16])
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            conv1 = conv_bn_layer(input, 8, 1, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 1, "conv2")
            conv3 = fluid.layers.conv2d_transpose(
                input=conv2, num_filters=16, filter_size=2, stride=2)
            predict = fluid.layers.fc(input=conv3, size=10, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
            avg_cost = fluid.layers.mean(cost)
            adam_optimizer.minimize(avg_cost)

        place = paddle.static.cpu_places()[0]
        exe = paddle.static.Executor(place)
        scope = paddle.static.global_scope()
        exe.run(startup_program, scope=scope)

        pruner = UnstructuredPruner(
            main_program, 'ratio', scope=scope, place=place)

        self._update_masks(pruner, 0.0)
        pruner.update_params()
        self._update_masks(pruner, 1.0)
        pruner.set_static_masks()
        sparsity_0 = pruner.total_sparse(main_program)
        x = np.random.random(size=(10, 3, 16, 16)).astype('float32')
        label = np.random.random(size=(10, 1)).astype('int64')
        loss_data, = exe.run(main_program,
                             feed={"image": x,
                                   "label": label},
                             fetch_list=[cost.name])
        sparsity_1 = UnstructuredPruner.total_sparse(main_program)
        pruner.update_params()
        sparsity_2 = UnstructuredPruner.total_sparse(main_program)
        print(sparsity_0, sparsity_1, sparsity_2)
        self.assertEqual(sparsity_0, 1.0)
        self.assertLess(abs(sparsity_2 - 1), 0.001)
        self.assertLess(sparsity_1, 1.0)


if __name__ == '__main__':
    unittest.main()

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
import os
sys.path.append("../")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.prune import Pruner
from static_case import StaticCase
from layers import conv_bn_layer
import random
from paddleslim.core import GraphWrapper
from paddleslim.prune.prune_worker import *


class TestPrune(StaticCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        # X: prune output channels
        # O: prune input channels
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[None, 3, 16, 16])
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                conv1 = conv_bn_layer(input, 8, 3, "conv1", act='relu')
                conv2 = conv_bn_layer(conv1, 8, 3, "conv2", act='leaky_relu')
                sum1 = conv1 + conv2
                conv3 = conv_bn_layer(sum1, 8, 3, "conv3", act='relu6')
                conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
                sum2 = conv4 + sum1
                conv5 = conv_bn_layer(sum2, 8, 3, "conv5")

                flag = fluid.layers.fill_constant([1], value=1, dtype='int32')
                rand_flag = paddle.randint(2, dtype='int32')
                cond = fluid.layers.less_than(x=flag, y=rand_flag)
                cond_output = fluid.layers.create_global_var(
                    shape=[1],
                    value=0.0,
                    dtype='float32',
                    persistable=False,
                    name='cond_output')

                def cond_block1():
                    cond_conv = conv_bn_layer(conv5, 8, 3, "conv_cond1_1")
                    fluid.layers.assign(input=cond_conv, output=cond_output)

                def cond_block2():
                    cond_conv1 = conv_bn_layer(conv5, 8, 3, "conv_cond2_1")
                    cond_conv2 = conv_bn_layer(cond_conv1, 8, 3, "conv_cond2_2")
                    fluid.layers.assign(input=cond_conv2, output=cond_output)

                fluid.layers.cond(cond, cond_block1, cond_block2)
                sum3 = fluid.layers.sum([sum2, cond_output])

                conv6 = conv_bn_layer(sum3, 8, 3, "conv6")
                sub1 = conv6 - sum3
                mult = sub1 * sub1
                conv7 = conv_bn_layer(
                    mult, 8, 3, "Depthwise_Conv7", groups=8, use_cudnn=False)
                floored = fluid.layers.floor(conv7)
                scaled = fluid.layers.scale(floored)
                concated = fluid.layers.concat([scaled, mult], axis=1)
                conv8 = conv_bn_layer(concated, 8, 3, "conv8")
                predict = fluid.layers.fc(input=conv8, size=10, act='softmax')
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
                avg_cost = fluid.layers.mean(cost)
                adam_optimizer.minimize(avg_cost)

        params = []
        for param in main_program.all_parameters():
            if 'conv' in param.name:
                params.append(param.name)
        #TODO: To support pruning convolution before fc layer.
        params.remove('conv8_weights')

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup_program)
        x = np.random.random(size=(10, 3, 16, 16)).astype('float32')
        label = np.random.random(size=(10, 1)).astype('int64')
        loss_data, = exe.run(main_program,
                             feed={"image": x,
                                   "label": label},
                             fetch_list=[cost.name])
        pruner = Pruner()
        main_program, _, _ = pruner.prune(
            main_program,
            fluid.global_scope(),
            params=params,
            ratios=[0.5] * len(params),
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None)

        loss_data, = exe.run(main_program,
                             feed={"image": x,
                                   "label": label},
                             fetch_list=[cost.name])


class TestUnsqueeze2(StaticCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[None, 3, 16, 16])
                conv1 = conv_bn_layer(input, 8, 3, "conv1", act='relu')
                out = paddle.unsqueeze(conv1, axis=[0])

        graph = GraphWrapper(main_program)
        cls = PRUNE_WORKER.get("unsqueeze2")
        out_var = graph.var(out.name)
        in_var = graph.var(conv1.name)
        op = out_var.inputs()[0]
        # pruning out
        pruned_params = []
        ret = {}
        worker = cls(op, pruned_params, {}, True)
        worker.prune(out_var, 2, [])
        for var, axis, _, _ in pruned_params:
            ret[var.name()] = axis
        self.assertTrue(ret == {
            'conv1_weights': 0,
            'conv1_bn_scale': 0,
            'conv1_bn_offset': 0,
            'conv1_bn_mean': 0,
            'conv1_bn_variance': 0
        })

        # pruning in
        pruned_params = []
        ret = {}
        worker = cls(op, pruned_params, {}, True)
        worker.prune(in_var, 1, [])
        for var, axis, _, _ in pruned_params:
            ret[var.name()] = axis
        self.assertTrue(ret == {})


class TestSqueeze2(StaticCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[1, 3, 16, 16])
                conv1 = conv_bn_layer(
                    input, 8, 3, "conv1", act='relu')  #[1, 8, 1, 1]
                out = paddle.squeeze(conv1)

        graph = GraphWrapper(main_program)
        cls = PRUNE_WORKER.get("squeeze2")
        out_var = graph.var(out.name)
        in_var = graph.var(conv1.name)
        op = out_var.inputs()[0]
        # pruning out
        pruned_params = []
        ret = {}
        worker = cls(op, pruned_params, {}, True)
        worker.prune(out_var, 0, [])
        for var, axis, _, _ in pruned_params:
            ret[var.name()] = axis
        self.assertTrue(ret == {
            'conv1_weights': 0,
            'conv1_bn_scale': 0,
            'conv1_bn_offset': 0,
            'conv1_bn_mean': 0,
            'conv1_bn_variance': 0
        })

        # pruning in
        pruned_params = []
        ret = {}
        worker = cls(op, pruned_params, {}, True)
        worker.prune(in_var, 1, [])
        for var, axis, _, _ in pruned_params:
            ret[var.name()] = axis
        self.assertTrue(ret == {})


class TestUnsupportAndDefault(StaticCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[1, 3, 16, 16])
                conv1 = conv_bn_layer(
                    input, 8, 3, "conv1", act='relu')  #[1, 8, 1, 1]
                # hit default pruning worker
                cast1 = paddle.cast(conv1, dtype="int32")
                # hit unsupported pruning worker
                out = paddle.reshape(cast1, shape=[1, -1])

        graph = GraphWrapper(main_program)
        cls = PRUNE_WORKER.get("conv2d")
        in_var = graph.var("conv1_weights")
        op = in_var.outputs()[0]
        # pruning input of conv op
        pruned_params = []
        ret = {}
        os.environ['OPS_UNSUPPORTED'] = "reshape2"
        worker = cls(op, pruned_params, {}, True)
        hit_unsupported_op = False
        try:
            worker.prune(in_var, 0, [])
        except UnsupportOpError as e:
            hit_unsupported_op = True
            print(e)
        self.assertTrue(hit_unsupported_op)


class TestConv2d(StaticCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[1, 3, 16, 16])

                conv1 = conv_bn_layer(
                    input, 6, 3, "conv1", groups=1, bias=True, act='relu')

        graph = GraphWrapper(main_program)
        cls = PRUNE_WORKER.get("conv2d")
        weight_var = graph.var("conv1_weights")
        in_var = graph.var("image")
        op = in_var.outputs()[0]
        out_var = op.outputs("Output")[0]
        # pruning weights of conv op
        pruned_params = []
        ret = {}
        worker = cls(op, pruned_params, {}, True)
        worker.prune(weight_var, 0, [])
        worker.prune(weight_var, 1, [])
        for var, axis, _, _ in pruned_params:
            if var.name() not in ret:
                ret[var.name()] = []
            ret[var.name()].append(axis)
        self.assertTrue(ret == {
            'conv1_weights': [0, 1],
            'conv1_out.b_0': [0],
            'conv1_bn_scale': [0],
            'conv1_bn_offset': [0],
            'conv1_bn_mean': [0],
            'conv1_bn_variance': [0]
        })
        # pruning out of conv op
        pruned_params = []
        ret = {}
        worker = cls(op, pruned_params, visited={}, skip_stranger=True)
        worker.prune(out_var, 1, [])
        for var, axis, _, _ in pruned_params:
            if var.name() not in ret:
                ret[var.name()] = []
            ret[var.name()].append(axis)
        self.assertTrue(ret == {'conv1_weights': [0]})


class TestPruneWorker(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.create_graph()
        self.cases = []
        self.set_cases()

    def define_layer(self, input):
        pass

    def set_cases(self):
        pass

    def create_graph(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with paddle.fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[8, 8, 16, 16])
                self.define_layer(input)
        self.graph = GraphWrapper(main_program)
        self.in_var = self.graph.var(self.input.name)
        self.out_var = self.graph.var(self.output.name)
        self.op = self.in_var.outputs()[0]

    def check_in_out(self):
        cls = PRUNE_WORKER.get(self.op.type())
        if cls is None:
            cls = PRUNE_WORKER.get("default_worker")

        # pruning input of conv op
        for _var, _axis, _ret in self.cases:
            pruned_params = []
            ret = {}
            worker = cls(self.op, pruned_params, visited={}, skip_stranger=True)
            try:
                worker.prune(_var, _axis, [])
            except UnsupportOpError as e:
                print(e)
                continue
            for var, axis, _, _ in pruned_params:
                if var.name() not in ret:
                    ret[var.name()] = []
                ret[var.name()].append(axis)
            self.assertTrue(ret == _ret)


class TestConv2dTranspose(TestPruneWorker):
    def define_layer(self, input):
        self.input = input
        conv1 = paddle.static.nn.conv2d_transpose(
            input, 6, 16, 3, name="conv1", bias_attr=False)
        self.output = conv1
        return conv1

    def set_cases(self):
        self.cases.append((self.in_var, 1, {'conv1.w_0': [0]}))
        self.cases.append((self.out_var, 1, {'conv1.w_0': [1]}))

    def test_prune(self):
        self.check_in_out()


class TestElementwiseMul(TestPruneWorker):
    def define_layer(self, input):
        conv1 = paddle.static.nn.conv2d(
            input, 3, 3, name="conv1", bias_attr=False)
        conv2 = paddle.static.nn.conv2d(
            input, 3, 3, name="conv2", bias_attr=False)
        self.input = conv1
        out = conv1 * conv2
        conv3 = paddle.static.nn.conv2d(
            out, 3, 3, name="conv3", bias_attr=False)
        self.output = out

    def set_cases(self):
        self.cases.append((self.in_var, 1, {
            'conv2.tmp_0': [1],
            'conv2.w_0': [0],
            'conv3.w_0': [1]
        }))
        self.cases.append((self.out_var, 1, {
            'conv1.w_0': [0],
            'conv2.tmp_0': [1],
            'conv2.w_0': [0]
        }))

    def test_prune(self):
        self.check_in_out()


class TestActivation(TestPruneWorker):
    def __init__(self, methodName="test_prune",
                 op=paddle.nn.functional.sigmoid):
        super(TestActivation, self).__init__(methodName)
        self.act = op

    def define_layer(self, input):
        conv1 = paddle.static.nn.conv2d(
            input, 3, 3, name="conv1", bias_attr=False)
        self.input = conv1
        tmp = self.act(conv1)
        self.output = tmp
        conv2 = paddle.static.nn.conv2d(
            tmp, 3, 3, name="conv2", bias_attr=False)

    def set_cases(self):
        self.cases.append((self.in_var, 1, {'conv2.w_0': [1]}))
        self.cases.append((self.out_var, 1, {
            'conv1.w_0': [0],
            'conv2.w_0': [1]
        }))

    def test_prune(self):
        self.check_in_out()


suite = unittest.TestSuite()
suite.addTest(TestActivation(op=paddle.fluid.layers.resize_bilinear))
suite.addTest(TestActivation(op=paddle.fluid.layers.resize_nearest))
suite.addTest(TestActivation(op=paddle.floor))
suite.addTest(TestActivation(op=paddle.scale))
suite.addTest(
    TestActivation(op=paddle.fluid.layers.nn.uniform_random_batch_size_like))


class TestDepthwiseConv2d(TestPruneWorker):
    def __init__(self, methodName="test_prune"):
        super(TestDepthwiseConv2d, self).__init__(methodName)

    def define_layer(self, input):
        self.input = input
        conv1 = paddle.static.nn.conv2d(
            input,
            input.shape[1],
            3,
            groups=input.shape[1],
            name="conv1",
            bias_attr=False)
        self.output = conv1

    def set_cases(self):
        weight_var = self.graph.var('conv1.w_0')
        self.cases.append((self.in_var, 1, {'conv1.w_0': [0, 1]}))
        self.cases.append((self.out_var, 1, {'conv1.w_0': [0, 1]}))
        self.cases.append((weight_var, 0, {'conv1.w_0': [1]}))

    def test_prune(self):
        self.check_in_out()


class TestMul(TestPruneWorker):
    def __init__(self, methodName="test_prune"):
        super(TestMul, self).__init__(methodName)

    def define_layer(self, input):
        x = fluid.data(name="x", shape=[1, 4, 3, 3])
        y = fluid.data(name="y", shape=[36, 7])
        self.input = x
        out = paddle.fluid.layers.mul(x, y)
        self.output = out

    def set_cases(self):
        self.cases.append((self.in_var, 1, {'y': [0]}))

    def test_prune(self):
        self.check_in_out()


class TestMatmul(TestPruneWorker):
    def __init__(self, methodName="test_prune"):
        super(TestMatmul, self).__init__(methodName)

    def define_layer(self, input):
        x = fluid.data(name="x", shape=[6, 8])
        y = fluid.data(name="y", shape=[8, 7])
        self.input = x
        out = paddle.matmul(x, y)
        self.output = out

    def set_cases(self):
        self.cases.append((self.in_var, 1, {'y': [0]}))
        self.cases.append((self.out_var, 0, {'x': [0]}))
        self.cases.append((self.out_var, 1, {'y': [1]}))

    def test_prune(self):
        self.check_in_out()


class TestSplit(TestPruneWorker):
    def define_layer(self, input):
        self.input = input
        split1 = paddle.split(input, num_or_sections=2, axis=1, name=None)
        self.output = split1[0]

    def set_cases(self):
        self.cases.append((self.in_var, 1, {}))
        self.cases.append((self.in_var, 0, {}))
        self.cases.append((self.out_var, 1, {}))
        self.cases.append((self.out_var, 0, {}))

    def test_prune(self):
        self.check_in_out()


class TestMomentum(TestPruneWorker):
    def define_layer(self, input):
        self.input = input
        conv1 = paddle.static.nn.conv2d(
            input, 3, 8, name="conv1", bias_attr=False)
        self.output = conv1
        out = paddle.mean(conv1)
        opt = paddle.optimizer.Momentum()
        opt.minimize(out)

    def set_cases(self):
        weight_var = self.graph.var('conv1.w_0')
        self.cases.append((weight_var, 0, {
            'conv1.w_0': [0],
            'conv1.w_0_velocity_0': [0]
        }))

    def test_prune(self):
        self.check_in_out()


class TestAdam(TestPruneWorker):
    def define_layer(self, input):
        self.input = input
        conv1 = paddle.static.nn.conv2d(
            input, 3, 8, name="conv1", bias_attr=False)
        self.output = conv1
        out = paddle.mean(conv1)
        opt = paddle.optimizer.Adam()
        opt.minimize(out)

    def set_cases(self):
        weight_var = self.graph.var('conv1.w_0')
        self.cases.append((weight_var, 0, {
            'conv1.w_0': [0],
            'conv1.w_0_moment1_0': [0],
            'conv1.w_0_moment2_0': [0]
        }))

    def test_prune(self):
        self.check_in_out()


class TestAffineChannel(TestPruneWorker):
    def __init__(self, methodName="test_prune"):
        super(TestAffineChannel, self).__init__(methodName)

    def define_layer(self, input):
        conv1 = paddle.static.nn.conv2d(
            input, 3, 8, name="conv1", bias_attr=False)

        self.input = conv1
        scale = fluid.data(name="scale", shape=[conv1.shape[1]])
        bias = fluid.data(name="bias", shape=[conv1.shape[1]])
        out = paddle.fluid.layers.affine_channel(conv1, scale=scale, bias=bias)
        self.output = out

    def set_cases(self):
        self.cases.append((self.in_var, 1, {'scale': [0], 'bias': [0]}))
        self.cases.append((self.out_var, 1, {
            'conv1.w_0': [0],
            'scale': [0],
            'bias': [0]
        }))

    def test_prune(self):
        self.check_in_out()


if __name__ == '__main__':
    unittest.main()

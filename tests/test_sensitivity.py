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
import numpy
import paddle
import paddle.fluid as fluid
from static_case import StaticCase
from paddleslim.prune import sensitivity, merge_sensitive, load_sensitivities
from layers import conv_bn_layer


class TestSensitivity(StaticCase):
    def test_sensitivity(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 1, 28, 28])
            label = fluid.data(name="label", shape=[None, 1], dtype="int64")
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")
            out = fluid.layers.fc(conv6, size=10, act='softmax')
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        eval_program = main_program.clone(for_test=True)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup_program)

        val_reader = paddle.fluid.io.batch(
            paddle.dataset.mnist.test(), batch_size=128)

        def eval_func(program):
            feeder = fluid.DataFeeder(
                feed_list=['image', 'label'], place=place, program=program)
            acc_set = []
            for data in val_reader():
                acc_np = exe.run(program=program,
                                 feed=feeder.feed(data),
                                 fetch_list=[acc_top1])
                acc_set.append(float(acc_np[0]))
            acc_val_mean = numpy.array(acc_set).mean()
            print("acc_val_mean: {}".format(acc_val_mean))
            return acc_val_mean

        def eval_func_for_args(args):
            program = args[0]
            feeder = fluid.DataFeeder(
                feed_list=['image', 'label'], place=place, program=program)
            acc_set = []
            for data in val_reader():
                acc_np = exe.run(program=program,
                                 feed=feeder.feed(data),
                                 fetch_list=[acc_top1])
                acc_set.append(float(acc_np[0]))
            acc_val_mean = numpy.array(acc_set).mean()
            print("acc_val_mean: {}".format(acc_val_mean))
            return acc_val_mean

        sensitivity(
            eval_program,
            place, ["conv4_weights"],
            eval_func,
            sensitivities_file="./sensitivities_file_0",
            pruned_ratios=[0.1, 0.2])

        sensitivity(
            eval_program,
            place, ["conv4_weights"],
            eval_func,
            sensitivities_file="./sensitivities_file_1",
            pruned_ratios=[0.3, 0.4])

        params_sens = sensitivity(
            eval_program,
            place, ["conv4_weights"],
            eval_func_for_args,
            eval_args=[eval_program],
            sensitivities_file="./sensitivites_file_params",
            pruned_ratios=[0.1, 0.2, 0.3, 0.4])

        sens_0 = load_sensitivities('./sensitivities_file_0')
        sens_1 = load_sensitivities('./sensitivities_file_1')
        sens = merge_sensitive([sens_0, sens_1])
        origin_sens = sensitivity(
            eval_program,
            place, ["conv4_weights"],
            eval_func,
            sensitivities_file="./sensitivities_file_2",
            pruned_ratios=[0.1, 0.2, 0.3, 0.4])
        self.assertTrue(params_sens == origin_sens)
        self.assertTrue(sens == origin_sens)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from paddleslim.dist import merge, l2_loss
from layers import conv_bn_layer


class TestL2Loss(unittest.TestCase):
    def test_l2_loss(self):
        student_main = fluid.Program()
        student_startup = fluid.Program()
        with fluid.program_guard(student_main, student_startup):
            input = fluid.data(name="image", shape=[None, 3, 224, 224])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            student_predict = conv1 + conv2

        teacher_main = fluid.Program()
        teacher_startup = fluid.Program()
        with fluid.program_guard(teacher_main, teacher_startup):
            input = fluid.data(name="image", shape=[None, 3, 224, 224])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            teacher_predict = conv_bn_layer(conv5, 8, 3, "conv6")

        place = fluid.CPUPlace()
        data_name_map = {'image': 'image'}
        merge(teacher_main, student_main, data_name_map, place)
        merged_ops = []
        for block in student_main.blocks:
            for op in block.ops:
                merged_ops.append(op.type)
        with fluid.program_guard(student_main):
            distill_loss = l2_loss('teacher_conv6_bn_output.tmp_2',
                                   'conv2_bn_output.tmp_2', student_main)
        loss_ops = []
        for block in student_main.blocks:
            for op in block.ops:
                loss_ops.append(op.type)
        self.assertTrue(set(merged_ops).difference(set(loss_ops)) == set())
        self.assertTrue(
            set(loss_ops).difference(set(merged_ops)) ==
            {'reduce_mean', 'square', 'elementwise_sub'})


if __name__ == '__main__':
    unittest.main()

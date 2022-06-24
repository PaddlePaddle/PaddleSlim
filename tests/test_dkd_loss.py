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
import paddle
from paddleslim.dist import merge, dkd
from layers import conv_bn_layer
from static_case import StaticCase


class TestFSPLoss(StaticCase):
    def test_dkd_loss(self):
        input = paddle.static.data(name="image", shape=[None, 3, 224, 224])
        conv1 = conv_bn_layer(input, 8, 3, "conv1")
        conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
        student_predict = conv1 + conv2
        student_predict = paddle.fluid.layers.fc(student_predict, size=10)

        teacher_main = paddle.static.Program()
        teacher_startup = paddle.static.Program()
        with paddle.static.program_guard(teacher_main, teacher_startup):
            input = paddle.static.data(name="image", shape=[None, 3, 224, 224])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            teacher_predict = conv_bn_layer(conv5, 8, 3, "conv6")
            teacher_predict = paddle.fluid.layers.fc(teacher_predict, size=10)

        place = paddle.CPUPlace()
        data_name_map = {'image': 'image'}
        merge(teacher_main,
              paddle.static.default_main_program(), data_name_map, place)

        merged_ops = []
        for block in paddle.static.default_main_program().blocks:
            for op in block.ops:
                merged_ops.append(op.type)

        distill_loss = dkd("teacher_" + (teacher_predict.name),
                           student_predict.name)
        loss_ops = []
        for block in paddle.static.default_main_program().blocks:
            for op in block.ops:
                loss_ops.append(op.type)
        self.assertTrue(set(merged_ops).difference(set(loss_ops)) == set())
        self.assertTrue(
            set(loss_ops).difference(set(merged_ops)) == {
                'kldiv_loss', 'assign', 'scale', 'concat', 'reduce_sum',
                'equal', 'softmax', 'reduce_mean', 'cast', 'elementwise_mul',
                'top_k_v2'
            })


if __name__ == '__main__':
    unittest.main()

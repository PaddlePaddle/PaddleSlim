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
import paddle
sys.path.append("../")
import paddleslim.quant as quant
import unittest

from static_case import StaticCase


class TestQuantEmbedding(StaticCase):
    def test_quant_embedding(self):
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program):
            input_word = paddle.static.data(
                name="input_word", shape=[None, 1], dtype='int64')
            param_attr = paddle.ParamAttr(
                name='emb',
                initializer=paddle.nn.initializer.Uniform(-0.005, 0.005))
            weight = train_program.global_block().create_parameter(
                (100, 128), attr=param_attr, dtype="float32")

            input_emb = paddle.nn.functional.embedding(
                x=input_word, weight=weight, sparse=True)

        infer_program = train_program.clone(for_test=True)

        use_gpu = True
        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        quant_program = quant.quant_embedding(infer_program, place)


if __name__ == '__main__':
    unittest.main()

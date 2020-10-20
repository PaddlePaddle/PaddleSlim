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
import paddle.fluid as fluid
import paddleslim.quant as quant
import unittest

from static_case import StaticCase


class TestQuantEmbedding(StaticCase):
    def test_quant_embedding(self):
        train_program = fluid.Program()
        with fluid.program_guard(train_program):
            input_word = fluid.data(
                name="input_word", shape=[None, 1], dtype='int64')
            input_emb = fluid.embedding(
                input=input_word,
                is_sparse=False,
                size=[100, 128],
                param_attr=fluid.ParamAttr(
                    name='emb',
                    initializer=fluid.initializer.Uniform(-0.005, 0.005)))

        infer_program = train_program.clone(for_test=True)

        use_gpu = True
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        quant_program = quant.quant_embedding(infer_program, place)


if __name__ == '__main__':
    unittest.main()

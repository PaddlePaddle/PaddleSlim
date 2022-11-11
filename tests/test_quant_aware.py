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
import paddle
from paddleslim.quant import quant_aware, convert
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
import numpy as np

class TestQuantAwareCase2(StaticCase):
    def test_accuracy(self):
        self.image = paddle.static.data(
            name='image', shape=[None, 1, 28, 28], dtype='float32')
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        model = MobileNet()
        self.out = model.net(input=self.image, class_dim=10)
        cost = paddle.nn.functional.loss.cross_entropy(input=self.out, label=label)
        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.metric.accuracy(input=self.out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=self.out, label=label, k=5)
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=paddle.regularizer.L2Decay(4e-5))
        optimizer.minimize(avg_cost)
        main_prog = paddle.static.default_main_program()
        val_prog = main_prog.clone(for_test=True)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        config = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
            'onnx_format': True
        }
        quant_train_prog = quant_aware(main_prog, place, config, for_test=False)
        
        
        quant_eval_prog = quant_aware(val_prog, place, config, for_test=True)
        print('Apply quantization passes for training')
        
        quant_eval_prog = convert(
            quant_eval_prog, place, config)

        paddle.fluid.io.save_inference_model(
            dirname='infer2',
            feeded_var_names=[self.image.name],
            target_vars=[self.out],
            executor=exe,
            main_program=quant_eval_prog,
            model_filename='infer2/model',
            params_filename='infer2/params')


if __name__ == '__main__':
    unittest.main()

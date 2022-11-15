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
import tempfile
import paddle
from paddleslim.quant import quant_post_static
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
import numpy as np
from paddleslim.quant import quant_recon_static


class ReconPTQ(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ReconPTQ, self).__init__(*args, **kwargs)
        paddle.enable_static()
        self.tmpdir = tempfile.TemporaryDirectory(prefix="test_")
        self._gen_model()

    def _gen_model(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            image = paddle.static.data(
                name='image', shape=[None, 1, 28, 28], dtype='float32')
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64')
            model = MobileNet()
            out = model.net(input=image, class_dim=10)
            cost = paddle.nn.functional.loss.cross_entropy(
                input=out, label=label)
            avg_cost = paddle.mean(x=cost)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            val_program = main_program.clone(for_test=True)
            optimizer = paddle.optimizer.Momentum(
                momentum=0.9,
                learning_rate=0.01,
                weight_decay=paddle.regularizer.L2Decay(4e-5))
            optimizer.minimize(avg_cost)
        exe.run(startup_program)

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform)
        self.train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[image, label],
            drop_last=True,
            batch_size=64,
            return_list=False)
        valid_loader = paddle.io.DataLoader(
            test_dataset,
            places=place,
            feed_list=[image, label],
            batch_size=64,
            return_list=False)

        def sample_generator_creator():
            def __reader__():
                for data in test_dataset:
                    image, label = data
                    yield image, label

            return __reader__

        def train(program):
            iter = 0
            for data in self.train_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print(
                        'train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                        format(iter, cost, top1, top5))

        train(main_program)
        paddle.fluid.io.save_inference_model(
            dirname=self.tmpdir.name,
            feeded_var_names=[image.name],
            target_vars=[out],
            main_program=val_program,
            executor=exe,
            model_filename='model.pdmodel',
            params_filename='params.pdiparams')
        print(f"saved infer model to [{self.tmpdir.name}]")
        self.data_loader = sample_generator_creator()

    def __del__(self):
        self.tmpdir.cleanup()


class TestReconRegion(ReconPTQ):
    def __init__(self, *args, **kwargs):
        super(TestReconRegion, self).__init__(*args, **kwargs)

    def test_qdrop_region(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        quant_recon_static(
            exe,
            self.tmpdir.name,
            quantize_model_path='output_region',
            sample_generator=self.data_loader,
            model_filename='model.pdmodel',
            params_filename='params.pdiparams',
            batch_nums=1,
            epochs=1,
            algo='abs_max',
            regions=None,
            region_weights_names=None,
            recon_level='region-wise',
            simulate_activation_quant=True)


class TestReconLayer(ReconPTQ):
    def __init__(self, *args, **kwargs):
        super(TestReconLayer, self).__init__(*args, **kwargs)

    def test_qdrop_layer(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        quant_recon_static(
            exe,
            self.tmpdir.name,
            quantize_model_path='output_layer',
            sample_generator=self.data_loader,
            model_filename='model.pdmodel',
            params_filename='params.pdiparams',
            batch_nums=1,
            epochs=1,
            algo='KL',
            regions=None,
            region_weights_names=None,
            recon_level='layer-wise',
            simulate_activation_quant=True,
            bias_correction=True)


if __name__ == '__main__':
    unittest.main()

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

import numpy as np
import sys
sys.path.append("../../")
import unittest
import logging
import paddle

from paddleslim.dygraph.quant import QAT

_logger = paddle.fluid.log_helper.get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativeLenet(paddle.nn.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(ImperativeLenet, self).__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1),
            paddle.nn.AvgPool2D(
                kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0),
            paddle.nn.AvgPool2D(
                kernel_size=2, stride=2))

        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=400, out_features=120),
            paddle.nn.Linear(
                in_features=120, out_features=84),
            paddle.nn.Linear(
                in_features=84, out_features=num_classes), )

    def forward(self, inputs):
        x = self.features(inputs)

        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


class TestQAT(unittest.TestCase):
    """
    QAT = quantization-aware training
    This test case uses defualt quantization config, weight_quantize_type 
    is channel_wise_abs_max
    """

    def set_seed(self):
        seed = 1
        np.random.seed(seed)
        paddle.static.default_main_program().random_seed = seed
        paddle.static.default_startup_program().random_seed = seed

    def prepare(self):
        self.quanter = QAT()

    def test_qat_acc(self):
        self.prepare()
        self.set_seed()

        fp32_lenet = ImperativeLenet()

        place = paddle.CUDAPlace(0) \
            if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

        transform = paddle.vision.transforms.Compose([
            paddle.vision.transforms.Transpose(),
            paddle.vision.transforms.Normalize([127.5], [127.5])
        ])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        val_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform)

        train_reader = paddle.io.DataLoader(
            train_dataset,
            drop_last=True,
            places=place,
            batch_size=64,
            return_list=True)
        test_reader = paddle.io.DataLoader(
            val_dataset, places=place, batch_size=64, return_list=True)

        def train(model):
            adam = paddle.optimizer.Adam(
                learning_rate=0.0001, parameters=model.parameters())
            epoch_num = 1
            for epoch in range(epoch_num):
                model.train()
                for batch_id, data in enumerate(train_reader):
                    img = paddle.to_tensor(data[0])
                    label = paddle.to_tensor(data[1])
                    img = paddle.reshape(img, [-1, 1, 28, 28])
                    label = paddle.reshape(label, [-1, 1])

                    out = model(img)
                    acc = paddle.metric.accuracy(out, label)
                    loss = paddle.nn.functional.loss.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    model.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))

        def test(model):
            model.eval()
            avg_acc = [[], []]
            for batch_id, data in enumerate(test_reader):
                img = paddle.to_tensor(data[0])
                img = paddle.reshape(img, [-1, 1, 28, 28])
                label = paddle.to_tensor(data[1])
                label = paddle.reshape(label, [-1, 1])

                out = model(img)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
                avg_acc[0].append(acc_top1.numpy())
                avg_acc[1].append(acc_top5.numpy())
                if batch_id % 100 == 0:
                    _logger.info(
                        "Test | step {}: acc1 = {:}, acc5 = {:}".format(
                            batch_id, acc_top1.numpy(), acc_top5.numpy()))

            _logger.info("Test | Average: acc_top1 {}, acc_top5 {}".format(
                np.mean(avg_acc[0]), np.mean(avg_acc[1])))
            return np.mean(avg_acc[0]), np.mean(avg_acc[1])

        train(fp32_lenet)
        top1_1, top5_1 = test(fp32_lenet)

        fp32_lenet.__init__()
        quant_lenet = self.quanter.quantize(fp32_lenet)
        train(quant_lenet)
        top1_2, top5_2 = test(quant_lenet)
        self.quanter.save_quantized_model(
            quant_lenet,
            './tmp/qat',
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])

        # values before quantization and after quantization should be close
        _logger.info("Before quantization: top1: {}, top5: {}".format(top1_1,
                                                                      top5_1))
        _logger.info("After quantization: top1: {}, top5: {}".format(top1_2,
                                                                     top5_2))
        _logger.info("\n")

        diff = 0.002
        self.assertTrue(
            top1_1 - top1_2 < diff,
            msg="The acc of quant model is too lower than fp32 model")


class TestQATWithPACT(TestQAT):
    """
    This test case is for testing user defined quantization.
    """

    def prepare(self):
        quant_config = {'activation_preprocess_type': 'PACT', }
        self.quanter = QAT(config=quant_config)


if __name__ == '__main__':
    unittest.main()

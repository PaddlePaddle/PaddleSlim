import sys
sys.path.append("../../")
import logging
import numpy as np
import unittest
import paddle
import paddle.nn as nn
from paddle.vision.models import MobileNetV1
import paddle.vision.transforms as T
from paddleslim.dygraph.dist import Distill, config2yaml
from paddleslim.common.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeDistill(unittest.TestCase):
    def setUp(self):
        self.s_model, self.t_model = self.prepare_model()
        self.t_model.eval()
        self.distill_configs = self.prepare_config()

    def prepare_model(self):
        return MobileNetV1(), MobileNetV1()

    def prepare_config(self):
        self.convert_fn = False
        distill_configs = [{
            'loss_function': 'MSELoss',
            'layers': [
                {
                    "layers_name": ["conv1", "conv1"]
                },
                {
                    "layers_name": ["conv2_2", "conv2_2"]
                },
            ]
        }, {
            'loss_function': 'CELoss',
            'temperature': 1.0,
            'layers': [{
                "layers_name": ["fc", "fc"]
            }, ]
        }]
        return distill_configs

    def test_distill(self):
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])

        train_dataset = paddle.vision.datasets.Cifar10(
            mode='train', backend='cv2', transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode='test', backend='cv2', transform=transform)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        train_reader = paddle.io.DataLoader(
            train_dataset, drop_last=True, places=place, batch_size=64)
        test_reader = paddle.io.DataLoader(
            val_dataset, places=place, batch_size=64)

        def test(model):
            model.eval()
            avg_acc = [[], []]
            for batch_id, data in enumerate(test_reader):
                img = paddle.to_tensor(data[0])
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

            _logger.info("Test |Average: acc_top1 {}, acc_top5 {}".format(
                np.mean(avg_acc[0]), np.mean(avg_acc[1])))

        def train(model):
            adam = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=model.parameters())

            for batch_id, data in enumerate(train_reader):
                img = paddle.to_tensor(data[0])
                label = paddle.to_tensor(data[1])
                distill_loss, student_out, teacher_out = model(img)
                loss = paddle.nn.functional.loss.cross_entropy(student_out,
                                                               label)
                avg_loss = paddle.mean(loss)
                all_loss = avg_loss + distill_loss
                all_loss.backward()
                adam.step()
                adam.clear_grad()
                if batch_id % 100 == 0:
                    _logger.info("Train | At epoch {} step {}: loss = {:}".
                                 format(str(0), batch_id, all_loss.numpy()))
            test(self.s_model)
            self.s_model.train()

        distill_model = Distill(
            self.distill_configs,
            self.s_model,
            self.t_model,
            convert_fn=self.convert_fn)
        train(distill_model)


class TestImperativeDistillCase1(TestImperativeDistill):
    def prepare_model(self):
        class convbn(nn.Layer):
            def __init__(self):
                super(convbn, self).__init__()
                self.conv = nn.Conv2D(3, 3, 3, padding=1)
                self.bn = nn.BatchNorm(3)

            def forward(self, x):
                conv_out = self.conv(x)
                bn_out = self.bn(conv_out)
                return tuple([conv_out, bn_out])

        class Model(nn.Layer):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2D(3, 3, 3, padding=1)
                self.conv2 = convbn()
                self.conv3 = nn.Conv2D(3, 3, 3, padding=1)
                self.fc = nn.Linear(3072, 10)

            def forward(self, x):
                self.conv1_out = self.conv1(x)
                conv2_out = self.conv2(self.conv1_out)
                self.conv3_out = self.conv3(conv2_out[0])
                out = paddle.reshape(self.conv3_out, shape=[x.shape[0], -1])
                out = paddle.nn.functional.softmax(out)
                out = self.fc(out)
                return out

        return Model(), Model()

    def prepare_config(self):
        self.convert_fn = True
        distill_configs = [{
            'loss_function': 'MSELoss',
            'layers': [
                {
                    "layers_name": ["conv1", "conv1"],
                    'align_params': {
                        'align_type': '1x1conv',
                        'in_channel': 3,
                        'out_channel': 3
                    }
                },
                {
                    "layers_name": ["conv2", "conv3"],
                    'io': ["input", "output"],
                    'align_params': {
                        'align_type': '3x3conv',
                        'in_channel': 3,
                        'out_channel': 3
                    }
                },
                {
                    "layers_name": ["conv2", "conv3"],
                    'io': ["output", "output"],
                    'idx': [1, None],
                    'align_params': {
                        'align_type': '1x1conv+bn',
                        'in_channel': 3,
                        'out_channel': 3
                    }
                },
                {
                    "layers_name": ["conv2", "conv3"],
                    'io': ["output", "output"],
                    'idx': [1, None],
                    'align_params': {
                        'align_type': '3x3conv+bn',
                        'in_channel': 3,
                        'out_channel': 3,
                        'transpose_model': 'student'
                    }
                },
            ]
        }, {
            'loss_function': 'CELoss',
            'temperature': 1.0,
            'layers': [{
                "layers_name": ["fc", "fc"],
                'align_params': {
                    'align_type': 'linear',
                    'in_channel': 10,
                    'out_channel': 10,
                    'weight_init': {
                        'initializer': 'Normal',
                        'mean': 0.0,
                        'std': 0.02
                    },
                }
            }, ]
        }]
        config2yaml(distill_configs, 'test.yaml')
        return './test.yaml'


if __name__ == '__main__':
    unittest.main()

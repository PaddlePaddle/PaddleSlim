import sys
sys.path.append("../../")
import logging
import numpy as np
import unittest
import paddle
import paddle.nn as nn
from paddle.vision.models import MobileNetV1
import paddle.vision.transforms as T
from paddleslim.dygraph.dist import Distill, AdaptorBase
from paddleslim.common.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeDistill(unittest.TestCase):
    def setUp(self):
        self.s_model, self.t_model = self.prepare_model()
        self.t_model.eval()
        self.distill_configs = self.prepare_config()
        self.adaptor = self.prepare_adaptor()

    def prepare_model(self):
        return MobileNetV1(), MobileNetV1()

    def prepare_config(self):
        distill_configs = [{
            's_feature_idx': 0,
            't_feature_idx': 0,
            'feature_type': 'hidden',
            'loss_function': 'l2'
        }, {
            's_feature_idx': 1,
            't_feature_idx': 1,
            'feature_type': 'hidden',
            'loss_function': 'l2'
        }, {
            's_feature_idx': 0,
            't_feature_idx': 0,
            'feature_type': 'logits',
            'loss_function': 'l2'
        }, {
            's_feature_idx': 2,
            't_feature_idx': 2,
            'feature_type': 'hidden',
            'loss_function': 'l2',
            'align': True,
            'transpose_model': 'student',
            'align_type': ['3x3conv'],
            'in_channels': [256],
            'out_channels': [256]
        }]
        return distill_configs

    def prepare_adaptor(self):
        class Adaptor(AdaptorBase):
            def mapping_layers(self):
                mapping_layers = {}
                mapping_layers['hidden_0'] = 'conv1'
                mapping_layers['hidden_1'] = 'conv2_2'
                mapping_layers['hidden_2'] = 'conv3_2'
                mapping_layers['logits_0'] = 'fc'
                return mapping_layers

        return Adaptor

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
                student_out, teacher_out, distill_loss = model(img)
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

        distill_model = Distill(self.distill_configs, self.s_model,
                                self.t_model, self.adaptor, self.adaptor)
        train(distill_model)


class TestImperativeDistillCase1(TestImperativeDistill):
    def prepare_model(self):
        class Model(nn.Layer):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2D(3, 3, 3, padding=1)
                self.conv2 = nn.Conv2D(3, 3, 3, padding=1)
                self.conv3 = nn.Conv2D(3, 3, 3, padding=1)
                self.fc = nn.Linear(3072, 10)

            def forward(self, x):
                self.conv1_out = self.conv1(x)
                conv2_out = self.conv2(self.conv1_out)
                self.conv3_out = self.conv3(conv2_out)
                out = paddle.reshape(self.conv3_out, shape=[x.shape[0], -1])
                out = self.fc(out)
                return out

        return Model(), Model()

    def prepare_adaptor(self):
        class Adaptor(AdaptorBase):
            def mapping_layers(self):
                mapping_layers = {}
                mapping_layers['hidden_1'] = 'conv2'
                if self.add_tensor:
                    mapping_layers['hidden_0'] = self.model.conv1_out
                    mapping_layers['hidden_2'] = self.model.conv3_out
                return mapping_layers

        return Adaptor

    def prepare_config(self):
        distill_configs = [{
            's_feature_idx': 0,
            't_feature_idx': 0,
            'feature_type': 'hidden',
            'loss_function': 'l2'
        }, {
            's_feature_idx': 1,
            't_feature_idx': 2,
            'feature_type': 'hidden',
            'loss_function': 'l2'
        }]
        return distill_configs


if __name__ == '__main__':
    unittest.main()

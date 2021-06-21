import sys
sys.path.append("../../")
import logging
import numpy as np
import unittest
import paddle
from paddle.fluid.log_helper import get_logger
from paddle.vision.models import MobileNetV1
import paddle.vision.transforms as T
from paddleslim.dygraph.dist import Distill, DistillConfig, add_distill_hook

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeDistill(unittest.TestCase):
    def setUp(self):
        self.s_model = MobileNetV1()
        self.t_model = MobileNetV1()
        self.t_model.eval()

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

        def adaptor(model, mapping_layers):
            mapping_keys = ['hidden', 'logits']
            for k in mapping_keys:
                mapping_layers[k] = []
            add_distill_hook(model, mapping_layers, ['conv1'], ['hidden_0'])
            add_distill_hook(model, mapping_layers, ['conv2_2'], ['hidden_1'])
            add_distill_hook(model, mapping_layers, ['conv3_2', 'conv4_2'],
                             ['hidden_2', 'hidden_3'])
            add_distill_hook(model, mapping_layers, ['fc'], ['logits_0'])
            return mapping_layers

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
                learning_rate=0.001, parameters=distill_model.parameters())

            for batch_id, data in enumerate(train_reader):
                img = paddle.to_tensor(data[0])
                label = paddle.to_tensor(data[1])
                student_out, teacher_out, distill_loss = distill_model(img)
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
        }]

        distill_model = Distill(distill_configs, self.s_model, self.t_model,
                                adaptor, adaptor)
        train(distill_model)


if __name__ == '__main__':
    unittest.main()

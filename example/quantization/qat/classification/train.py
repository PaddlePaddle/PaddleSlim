# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import paddle
import time
import random
import numpy as np
from paddleslim.common import get_logger
from paddle.quantization import QuantConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddle.quantization.quanters.abs_max import FakeQuanterWithAbsMaxObserverLayer
from paddleslim.quant.quanters import PACTQuanter
from paddle.quantization import QAT

sys.path.append(os.path.join(os.path.dirname("__file__")))
from optimizer import create_optimizer
from args import parse_args
from args import SUPPORT_MODELS
_logger = get_logger(__name__, level=logging.INFO)


def compress(args):
    num_workers = 4
    shuffle = True
    if args.ce_test:
        # set seed
        seed = 111
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        num_workers = 0
        shuffle = False

    if args.data == "cifar10":
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = paddle.vision.datasets.Cifar10(
            mode="train", backend="cv2", transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode="test", backend="cv2", transform=transform)
        class_dim = 10
        image_shape = [3, 32, 32]
        pretrain = False
        args.total_images = 50000
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1

    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    # model definition
    if use_data_parallel:
        paddle.distributed.init_parallel_env()

    pretrain = True if args.data == "imagenet" else False

    model = SUPPORT_MODELS[args.model](
        pretrained=pretrain, num_classes=class_dim)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        places=place,
        return_list=True,
        num_workers=num_workers)

    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        return_list=True,
        num_workers=num_workers)

    @paddle.no_grad()
    def test(epoch, net):
        net.eval()
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []

        eval_reader_cost = 0.0
        eval_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for data in valid_loader():
            eval_reader_cost += time.time() - reader_start
            image = data[0]
            label = data[1]
            if args.data == "cifar10":
                label = paddle.reshape(label, [-1, 1])

            eval_start = time.time()

            out = net(image)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            eval_run_cost += time.time() - eval_start
            batch_size = image.shape[0]
            total_samples += batch_size

            if batch_id % args.log_period == 0:
                log_period = 1 if batch_id == 0 else args.log_period
                _logger.info(
                    "Eval epoch[{}] batch[{}] - top1: {:.6f}; top5: {:.6f}; avg_reader_cost: {:.6f} s, avg_batch_cost: {:.6f} s, avg_samples: {}, avg_ips: {:.3f} images/s".
                    format(epoch, batch_id,
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), eval_reader_cost /
                           log_period, (eval_reader_cost + eval_run_cost) /
                           log_period, total_samples / log_period, total_samples
                           / (eval_reader_cost + eval_run_cost)))
                eval_reader_cost = 0.0
                eval_run_cost = 0.0
                total_samples = 0
            acc_top1_ns.append(np.mean(acc_top1.numpy()))
            acc_top5_ns.append(np.mean(acc_top5.numpy()))
            batch_id += 1
            reader_start = time.time()

        _logger.info(
            "Final eval epoch[{}] - acc_top1: {:.6f}; acc_top5: {:.6f}".format(
                epoch,
                np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))

    test(-1, model)

    ############################################################################################################
    # 1. quantization
    ############################################################################################################
    activation_quanter = PACTQuanter(FakeQuanterWithAbsMaxObserverLayer)
    weight_quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
    q_config = QuantConfig(activation=None, weight=None)
    q_config.add_type_config(
        [paddle.nn.Conv2D, paddle.nn.Linear],
        activation=activation_quanter,
        weight=weight_quanter)
    quanter = QAT(config=q_config)
    quant_model = quanter.quantize(model)

    opt, lr = create_optimizer(quant_model, trainer_num, args)

    if use_data_parallel:
        net = paddle.DataParallel(quant_model)

    def cross_entropy(input, target, ls_epsilon):
        if ls_epsilon > 0:
            if target.shape[-1] != class_dim:
                target = paddle.nn.functional.one_hot(target, class_dim)
            target = paddle.nn.functional.label_smooth(
                target, epsilon=ls_epsilon)
            target = paddle.reshape(target, shape=[-1, class_dim])
            input = -paddle.nn.functional.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = paddle.nn.functional.cross_entropy(input=input, label=target)
        avg_cost = paddle.mean(cost)
        return avg_cost

    def train(epoch, net):

        net.train()
        batch_id = 0

        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for data in train_loader():
            train_reader_cost += time.time() - reader_start

            image = data[0]
            label = data[1]
            if args.data == "cifar10":
                label = paddle.reshape(label, [-1, 1])

            train_start = time.time()
            out = net(image)
            avg_cost = cross_entropy(out, label, args.ls_epsilon)

            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
            avg_cost.backward()
            opt.step()
            opt.clear_grad()
            lr.step()

            loss_n = np.mean(avg_cost.numpy())
            acc_top1_n = np.mean(acc_top1.numpy())
            acc_top5_n = np.mean(acc_top5.numpy())

            train_run_cost += time.time() - train_start
            batch_size = image.shape[0]
            total_samples += batch_size

            if batch_id % args.log_period == 0:
                log_period = 1 if batch_id == 0 else args.log_period
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {:.6f}; top1: {:.6f}; top5: {:.6f}; avg_reader_cost: {:.6f} s, avg_batch_cost: {:.6f} s, avg_samples: {}, avg_ips: {:.3f} images/s".
                    format(epoch, batch_id,
                           lr.get_lr(), loss_n, acc_top1_n, acc_top5_n,
                           train_reader_cost / log_period, (
                               train_reader_cost + train_run_cost) / log_period,
                           total_samples / log_period, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            batch_id += 1
            reader_start = time.time()

    ############################################################################################################
    # train loop
    ############################################################################################################
    start_epoch = 0
    ck_info = args.checkpoints + "/checkpoints.info"
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    if os.path.isfile(ck_info):
        with open(ck_info, 'r') as f:
            start_epoch = int(f.readline()) + 1
        quant_model.load_dict(
            paddle.load(f"{args.checkpoints}/{start_epoch-1}.pdparams"))
        _logger.info(
            f"Load checkpoint from {args.checkpoints}/{start_epoch-1}.pdparams")
        test(start_epoch - 1, quant_model)

    for _epoch in range(start_epoch, args.num_epochs):
        train(_epoch, quant_model)
        acc1 = test(_epoch, quant_model)
        paddle.save(quant_model.state_dict(),
                    f"{args.checkpoints}/{_epoch}.pdparams")

        with open(ck_info, 'w') as f:
            f.write(str(_epoch))
        _logger.info(f"Save checkpoint to {args.checkpoints}/{_epoch}.pdparams")

    infer_model = quanter.convert(quant_model)

    dummy_input = paddle.static.InputSpec(
        shape=[None, 3, 224, 224], dtype='float32')
    paddle.jit.save(infer_model, args.infer_model, [dummy_input])
    _logger.info(f"Saved inference model to {args.infer_model}")


if __name__ == '__main__':

    args = parse_args()
    compress(args)

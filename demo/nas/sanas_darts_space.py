import os
import sys
sys.path.append('..')
import numpy as np
import argparse
import ast
import time
import argparse
import ast
import logging
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import paddle.static as static
from paddleslim.nas import SANAS
from paddleslim.common import get_logger
import darts_cifar10_reader as reader

_logger = get_logger(__name__, level=logging.INFO)

auxiliary = True
auxiliary_weight = 0.4
trainset_num = 50000
lr = 0.025
momentum = 0.9
weight_decay = 0.0003
drop_path_probility = 0.2


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def count_parameters_in_MB(all_params, prefix='model'):
    parameters_number = 0
    for param in all_params:
        if param.name.startswith(
                prefix) and param.trainable and 'aux' not in param.name:
            parameters_number += np.prod(param.shape)
    return parameters_number / 1e6


def create_data_loader(image_shape, is_train, args):
    image = static.data(
        name="image", shape=[None] + image_shape, dtype="float32")
    label = static.data(name="label", shape=[None, 1], dtype="int64")
    data_loader = paddle.io.DataLoader.from_generator(
        feed_list=[image, label],
        capacity=64,
        use_double_buffer=True,
        iterable=True)
    drop_path_prob = ''
    drop_path_mask = ''
    if is_train:
        drop_path_prob = static.data(
            name="drop_path_prob", shape=[args.batch_size, 1], dtype="float32")
        drop_path_mask = static.data(
            name="drop_path_mask",
            shape=[args.batch_size, 20, 4, 2],
            dtype="float32")

    return data_loader, image, label, drop_path_prob, drop_path_mask


def build_program(main_program, startup_program, image_shape, archs, args,
                  is_train):
    with static.program_guard(main_program, startup_program):
        data_loader, data, label, drop_path_prob, drop_path_mask = create_data_loader(
            image_shape, is_train, args)
        logits, logits_aux = archs(data, drop_path_prob, drop_path_mask,
                                   is_train, 10)
        top1 = paddle.metric.accuracy(input=logits, label=label, k=1)
        top5 = paddle.metric.accuracy(input=logits, label=label, k=5)
        loss = paddle.mean(F.softmax_with_cross_entropy(logits, label))

        if is_train:
            if auxiliary:
                loss_aux = paddle.mean(
                    F.softmax_with_cross_entropy(logits_aux, label))
                loss = loss + auxiliary_weight * loss_aux
            step_per_epoch = int(trainset_num / args.batch_size)
            learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
                lr, T_max=step_per_epoch * args.retain_epoch)
            optimizer = paddle.optimizer.Momentum(
                learning_rate,
                momentum,
                weight_decay=paddle.regularizer.L2Decay(weight_decay),
                grad_clip=nn.ClipGradByGlobalNorm(clip_norm=5.0))
            optimizer.minimize(loss)
            outs = [loss, top1, top5]
        else:
            outs = [loss, top1, top5]
    return outs, (data, label), data_loader


def train(main_prog, exe, epoch_id, train_loader, fetch_list, args):
    loss = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for step_id, data in enumerate(train_loader()):
        devices_num = len(data)
        if drop_path_probility > 0:
            feed = []
            for device_id in range(devices_num):
                image = data[device_id]['image']
                label = data[device_id]['label']
                drop_path_prob = np.array(
                    [[drop_path_probility * epoch_id / args.retain_epoch]
                     for i in range(args.batch_size)]).astype(np.float32)
                drop_path_mask = 1 - np.random.binomial(
                    1, drop_path_prob[0],
                    size=[args.batch_size, 20, 4, 2]).astype(np.float32)
                feed.append({
                    "image": image,
                    "label": label,
                    "drop_path_prob": drop_path_prob,
                    "drop_path_mask": drop_path_mask
                })
        else:
            feed = data
        loss_v, top1_v, top5_v = exe.run(
            main_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % 10 == 0:
            _logger.info(
                "Train Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[
                    0]))
    return top1.avg[0]


def valid(main_prog, exe, epoch_id, valid_loader, fetch_list, args):
    loss = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for step_id, data in enumerate(valid_loader()):
        loss_v, top1_v, top5_v = exe.run(
            main_prog, feed=data, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % 10 == 0:
            _logger.info(
                "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[
                    0]))
    return top1.avg[0]


def search(config, args, image_size, is_server=True):
    places = static.cuda_places() if args.use_gpu else static.cpu_places()
    place = places[0]
    if is_server:
        ### start a server and a client
        sa_nas = SANAS(
            config,
            server_addr=(args.server_address, args.port),
            search_steps=args.search_steps,
            is_server=True)
    else:
        ### start a client
        sa_nas = SANAS(
            config,
            server_addr=(args.server_address, args.port),
            init_temperature=init_temperature,
            is_server=False)

    image_shape = [3, image_size, image_size]
    for step in range(args.search_steps):
        archs = sa_nas.next_archs()[0]

        train_program = static.Program()
        test_program = static.Program()
        startup_program = static.Program()
        train_fetch_list, _, train_loader = build_program(
            train_program,
            startup_program,
            image_shape,
            archs,
            args,
            is_train=True)

        current_params = count_parameters_in_MB(
            train_program.global_block().all_parameters(), 'cifar10')
        _logger.info('step: {}, current_params: {}M'.format(step,
                                                            current_params))
        if current_params > float(3.77):
            continue

        test_fetch_list, _, test_loader = build_program(
            test_program,
            startup_program,
            image_shape,
            archs,
            args,
            is_train=False)
        test_program = test_program.clone(for_test=True)

        exe = static.Executor(place)
        exe.run(startup_program)

        train_reader = reader.train_valid(
            batch_size=args.batch_size, is_train=True, is_shuffle=True)
        test_reader = reader.train_valid(
            batch_size=args.batch_size, is_train=False, is_shuffle=False)

        train_loader.set_batch_generator(train_reader, places=place)
        test_loader.set_batch_generator(test_reader, places=place)

        build_strategy = static.BuildStrategy()
        train_compiled_program = static.CompiledProgram(
            train_program).with_data_parallel(
                loss_name=train_fetch_list[0].name,
                build_strategy=build_strategy)

        valid_top1_list = []
        for epoch_id in range(args.retain_epoch):
            train_top1 = train(train_compiled_program, exe, epoch_id,
                               train_loader, train_fetch_list, args)
            _logger.info("TRAIN: step: {}, Epoch {}, train_acc {:.6f}".format(
                step, epoch_id, train_top1))
            valid_top1 = valid(test_program, exe, epoch_id, test_loader,
                               test_fetch_list, args)
            _logger.info("TEST: Epoch {}, valid_acc {:.6f}".format(epoch_id,
                                                                   valid_top1))
            valid_top1_list.append(valid_top1)
        sa_nas.reward(float(valid_top1_list[-1] + valid_top1_list[-2]) / 2)


def final_test(config, args, image_size, token=None):
    assert token != None, "If you want to start a final experiment, you must input a token."
    places = static.cuda_places() if args.use_gpu else static.cpu_places()
    place = places[0]
    sa_nas = SANAS(
        config, server_addr=(args.server_address, args.port), is_server=True)

    image_shape = [3, image_size, image_size]
    archs = sa_nas.tokens2arch(token)[0]

    train_program = static.Program()
    test_program = static.Program()
    startup_program = static.Program()
    train_fetch_list, (data, label), train_loader = build_program(
        train_program, startup_program, image_shape, archs, args, is_train=True)

    current_params = count_parameters_in_MB(
        train_program.global_block().all_parameters(), 'cifar10')
    _logger.info('current_params: {}M'.format(current_params))
    test_fetch_list, _, test_loader = build_program(
        test_program, startup_program, image_shape, archs, args, is_train=False)
    test_program = test_program.clone(for_test=True)

    exe = static.Executor(place)
    exe.run(startup_program)

    train_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=True, is_shuffle=True)
    test_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=False, is_shuffle=False)

    train_loader.set_batch_generator(train_reader, places=place)
    test_loader.set_batch_generator(test_reader, places=place)

    build_strategy = static.BuildStrategy()
    train_compiled_program = static.CompiledProgram(
        train_program).with_data_parallel(
            loss_name=train_fetch_list[0].name, build_strategy=build_strategy)

    valid_top1_list = []
    for epoch_id in range(args.retain_epoch):
        train_top1 = train(train_compiled_program, exe, epoch_id, train_loader,
                           train_fetch_list, args)
        _logger.info("TRAIN: Epoch {}, train_acc {:.6f}".format(epoch_id,
                                                                train_top1))
        valid_top1 = valid(test_program, exe, epoch_id, test_loader,
                           test_fetch_list, args)
        _logger.info("TEST: Epoch {}, valid_acc {:.6f}".format(epoch_id,
                                                               valid_top1))
        valid_top1_list.append(valid_top1)

        output_dir = os.path.join('darts_output', str(epoch_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        static.save_inference_model(output_dir, [data], test_fetch_list, exe)


if __name__ == '__main__':

    paddle.enable_static()
    parser = argparse.ArgumentParser(
        description='SA NAS MobileNetV2 cifar10 argparase')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='Whether to use GPU in train/test model.')
    parser.add_argument(
        '--batch_size', type=int, default=96, help='batch size.')
    parser.add_argument(
        '--is_server',
        type=ast.literal_eval,
        default=True,
        help='Whether to start a server.')
    parser.add_argument(
        '--server_address', type=str, default="", help='server ip.')
    parser.add_argument('--port', type=int, default=8881, help='server port')
    parser.add_argument(
        '--retain_epoch', type=int, default=30, help='epoch for each token.')
    parser.add_argument('--token', type=int, nargs='+', help='final token.')
    parser.add_argument(
        '--search_steps',
        type=int,
        default=200,
        help='controller server number.')
    args = parser.parse_args()
    print(args)

    image_size = 32

    config = [('DartsSpace')]

    if args.token == None:
        search(config, args, image_size, is_server=args.is_server)
    else:
        final_test(config, args, image_size, token=args.token)

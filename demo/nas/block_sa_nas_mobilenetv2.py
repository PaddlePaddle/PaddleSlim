import sys
sys.path.append('..')
import numpy as np
import argparse
import ast
import logging
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
import paddle.static as static
from paddle import ParamAttr
from paddleslim.analysis import flops
from paddleslim.nas import SANAS
from paddleslim.common import get_logger
from optimizer import create_optimizer
import imagenet_reader

_logger = get_logger(__name__, level=logging.INFO)


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding='SAME',
                  num_groups=1,
                  act=None,
                  name=None,
                  use_cudnn=True):
    conv = static.nn.conv2d(
        input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=ParamAttr(name=name + '_weights'),
        bias_attr=False)
    bn_name = name + '_bn'
    return static.nn.batch_norm(
        input=conv,
        act=act,
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(name=bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')


def search_mobilenetv2_block(config, args, image_size):
    image_shape = [3, image_size, image_size]
    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    if args.data == 'cifar10':
        train_dataset = paddle.vision.datasets.Cifar10(
            mode='train', transform=transform, backend='cv2')
        val_dataset = paddle.vision.datasets.Cifar10(
            mode='test', transform=transform, backend='cv2')

    elif args.data == 'imagenet':
        train_dataset = imagenet_reader.ImageNetDataset(mode='train')
        val_dataset = imagenet_reader.ImageNetDataset(mode='val')

    places = static.cuda_places() if args.use_gpu else static.cpu_places()
    place = places[0]
    if args.is_server:
        sa_nas = SANAS(
            config,
            server_addr=(args.server_address, args.port),
            search_steps=args.search_steps,
            is_server=True)
    else:
        sa_nas = SANAS(
            config,
            server_addr=(args.server_address, args.port),
            search_steps=args.search_steps,
            is_server=False)

    for step in range(args.search_steps):
        archs = sa_nas.next_archs()[0]

        train_program = static.Program()
        test_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(train_program, startup_program):
            data_shape = [None] + image_shape
            data = static.data(name='data', shape=data_shape, dtype='float32')
            label = static.data(name='label', shape=[None, 1], dtype='int64')
            if args.data == 'cifar10':
                paddle.assign(paddle.reshape(label, [-1, 1]), label)
            train_loader = paddle.io.DataLoader(
                train_dataset,
                places=places,
                feed_list=[data, label],
                drop_last=True,
                batch_size=args.batch_size,
                return_list=False,
                shuffle=True,
                use_shared_memory=True,
                num_workers=4)
            val_loader = paddle.io.DataLoader(
                val_dataset,
                places=place,
                feed_list=[data, label],
                drop_last=False,
                batch_size=args.batch_size,
                return_list=False,
                shuffle=False)
            data = conv_bn_layer(
                input=data,
                num_filters=32,
                filter_size=3,
                stride=2,
                padding='SAME',
                act='relu6',
                name='mobilenetv2_conv1')
            data = archs(data)[0]
            data = conv_bn_layer(
                input=data,
                num_filters=1280,
                filter_size=1,
                stride=1,
                padding='SAME',
                act='relu6',
                name='mobilenetv2_last_conv')
            data = F.adaptive_avg_pool2d(
                data, output_size=[1, 1], name='mobilenetv2_last_pool')
            output = static.nn.fc(
                x=data,
                size=args.class_dim,
                weight_attr=ParamAttr(name='mobilenetv2_fc_weights'),
                bias_attr=ParamAttr(name='mobilenetv2_fc_offset'))

            softmax_out = F.softmax(output)
            cost = F.cross_entropy(softmax_out, label=label)
            avg_cost = paddle.mean(cost)
            acc_top1 = paddle.metric.accuracy(
                input=softmax_out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(
                input=softmax_out, label=label, k=5)
            test_program = train_program.clone(for_test=True)

            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                weight_decay=paddle.regularizer.L2Decay(1e-4))
            optimizer.minimize(avg_cost)

        current_flops = flops(train_program)
        print('step: {}, current_flops: {}'.format(step, current_flops))
        if current_flops > int(321208544):
            continue

        exe = static.Executor(place)
        exe.run(startup_program)

        build_strategy = static.BuildStrategy()
        train_compiled_program = static.CompiledProgram(
            train_program).with_data_parallel(
                loss_name=avg_cost.name, build_strategy=build_strategy)
        for epoch_id in range(args.retain_epoch):
            for batch_id, data in enumerate(train_loader()):
                fetches = [avg_cost.name]
                s_time = time.time()
                outs = exe.run(train_compiled_program,
                               feed=data,
                               fetch_list=fetches)[0]
                batch_time = time.time() - s_time
                if batch_id % 10 == 0:
                    _logger.info(
                        'TRAIN: steps: {}, epoch: {}, batch: {}, cost: {}, batch_time: {}ms'.
                        format(step, epoch_id, batch_id, outs[0], batch_time))

        reward = []
        for batch_id, data in enumerate(val_loader()):
            test_fetches = [avg_cost.name, acc_top1.name, acc_top5.name]
            batch_reward = exe.run(test_program,
                                   feed=data,
                                   fetch_list=test_fetches)
            reward_avg = np.mean(np.array(batch_reward), axis=1)
            reward.append(reward_avg)

            _logger.info(
                'TEST: step: {}, batch: {}, avg_cost: {}, acc_top1: {}, acc_top5: {}'.
                format(step, batch_id, batch_reward[0], batch_reward[1],
                       batch_reward[2]))

        finally_reward = np.mean(np.array(reward), axis=0)
        _logger.info(
            'FINAL TEST: avg_cost: {}, acc_top1: {}, acc_top5: {}'.format(
                finally_reward[0], finally_reward[1], finally_reward[2]))

        sa_nas.reward(float(finally_reward[1]))


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
        '--class_dim', type=int, default=1000, help='classify number.')
    parser.add_argument(
        '--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument(
        '--data',
        type=str,
        default='cifar10',
        choices=['cifar10', 'imagenet'],
        help='dataset name.')
    parser.add_argument(
        '--is_server',
        type=ast.literal_eval,
        default=True,
        help='Whether to start a server.')
    # nas args
    parser.add_argument(
        '--search_steps',
        type=int,
        default=100,
        help='controller server number.')
    parser.add_argument(
        '--server_address', type=str, default="", help='server ip.')
    parser.add_argument('--port', type=int, default=8881, help='server port')
    parser.add_argument(
        '--retain_epoch', type=int, default=5, help='epoch for each token.')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
    args = parser.parse_args()
    print(args)

    if args.data == 'cifar10':
        image_size = 32
    elif args.data == 'imagenet':
        image_size = 224
    else:
        raise NotImplementedError(
            'data must in [cifar10, imagenet], but received: {}'.format(
                args.data))

    # block mask means block number, 1 mean downsample, 0 means the size of feature map don't change after this block
    config_info = {'block_mask': [0, 1, 1, 1, 0]}
    config = [('MobileNetV2BlockSpace', config_info)]

    search_mobilenetv2_block(config, args, image_size)

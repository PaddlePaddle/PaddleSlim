import sys
sys.path.append('..')
import numpy as np
import argparse
import ast
import logging
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.analysis import flops
from paddleslim.nas import SANAS
from paddleslim.common import get_logger
from optimizer import create_optimizer
import imagenet_reader

_logger = get_logger(__name__, level=logging.INFO)

reduce_rate = 0.85
init_temperature = 10.24
max_flops = 321208544
server_address = ""
port = 8979
retain_epoch = 5


def create_data_loader(image_shape):
    data_shape = [None] + image_shape
    data = fluid.data(name='data', shape=data_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[data, label],
        capacity=1024,
        use_double_buffer=True,
        iterable=True)
    return data_loader, data, label


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding='SAME',
                  num_groups=1,
                  act=None,
                  name=None,
                  use_cudnn=True):
    conv = fluid.layers.conv2d(
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
    return fluid.layers.batch_norm(
        input=conv,
        act=act,
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(name=bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')


def search_mobilenetv2_block(config, args, image_size):
    image_shape = [3, image_size, image_size]
    if args.is_server:
        sa_nas = SANAS(
            config,
            server_addr=("", port),
            init_temperature=init_temperature,
            reduce_rate=reduce_rate,
            search_steps=args.search_steps,
            is_server=True)
    else:
        sa_nas = SANAS(
            config,
            server_addr=(server_address, port),
            init_temperature=init_temperature,
            reduce_rate=reduce_rate,
            search_steps=args.search_steps,
            is_server=False)

    for step in range(args.search_steps):
        archs = sa_nas.next_archs()[0]

        train_program = fluid.Program()
        test_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            train_loader, data, label = create_data_loader(image_shape)
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
            data = fluid.layers.pool2d(
                input=data,
                pool_size=7,
                pool_stride=1,
                pool_type='avg',
                global_pooling=True,
                name='mobilenetv2_last_pool')
            output = fluid.layers.fc(
                input=data,
                size=args.class_dim,
                param_attr=ParamAttr(name='mobilenetv2_fc_weights'),
                bias_attr=ParamAttr(name='mobilenetv2_fc_offset'))

            softmax_out = fluid.layers.softmax(input=output, use_cudnn=False)
            cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
            avg_cost = fluid.layers.mean(cost)
            acc_top1 = fluid.layers.accuracy(
                input=softmax_out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(
                input=softmax_out, label=label, k=5)
            test_program = train_program.clone(for_test=True)

            optimizer = fluid.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
            optimizer.minimize(avg_cost)

        current_flops = flops(train_program)
        print('step: {}, current_flops: {}'.format(step, current_flops))
        if current_flops > max_flops:
            continue

        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        if args.data == 'cifar10':
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.cifar.train10(cycle=False), buf_size=1024),
                batch_size=args.batch_size,
                drop_last=True)

            test_reader = paddle.batch(
                paddle.dataset.cifar.test10(cycle=False),
                batch_size=args.batch_size,
                drop_last=False)
        elif args.data == 'imagenet':
            train_reader = paddle.batch(
                imagenet_reader.train(),
                batch_size=args.batch_size,
                drop_last=True)
            test_reader = paddle.batch(
                imagenet_reader.val(),
                batch_size=args.batch_size,
                drop_last=False)

        test_loader, _, _ = create_data_loader(image_shape)
        train_loader.set_sample_list_generator(
            train_reader,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())
        test_loader.set_sample_list_generator(test_reader, places=place)

        build_strategy = fluid.BuildStrategy()
        train_compiled_program = fluid.CompiledProgram(
            train_program).with_data_parallel(
                loss_name=avg_cost.name, build_strategy=build_strategy)
        for epoch_id in range(retain_epoch):
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
        for batch_id, data in enumerate(test_loader()):
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
    config_info = {'block_mask': [0, 1, 1, 1, 1, 0, 1, 0]}
    config = [('MobileNetV2BlockSpace', config_info)]

    search_mobilenetv2_block(config, args, image_size)

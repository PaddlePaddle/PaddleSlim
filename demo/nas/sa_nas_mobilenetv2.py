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
port = 8909
retain_epoch = 5


def create_data_loader(image_shape):
    data_shape = [-1] + image_shape
    data = fluid.data(name='data', shape=data_shape, dtype='float32')
    label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[data, label],
        capacity=1024,
        use_double_buffer=True,
        iterable=True)
    return data_loader, data, label


def build_program(main_program,
                  startup_program,
                  image_shape,
                  archs,
                  args,
                  is_test=False):
    with fluid.program_guard(main_program, startup_program):
        data_loader, data, label = create_data_loader(image_shape)
        output = archs(data)
        output = fluid.layers.fc(
            input=output,
            size=args.class_dim,
            param_attr=ParamAttr(name='mobilenetv2_fc_weights'),
            bias_attr=ParamAttr(name='mobilenetv2_fc_offset'))

        softmax_out = fluid.layers.softmax(input=output, use_cudnn=False)
        cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)

        if is_test == False:
            optimizer = create_optimizer(args)
            optimizer.minimize(avg_cost)
    return data_loader, avg_cost, acc_top1, acc_top5


def search_mobilenetv2(config, args, image_size, is_server=True):
    if is_server:
        ### start a server and a client
        sa_nas = SANAS(
            config,
            server_addr=("", port),
            init_temperature=init_temperature,
            reduce_rate=reduce_rate,
            search_steps=args.search_steps,
            is_server=True)
    else:
        ### start a client
        sa_nas = SANAS(
            config,
            server_addr=(server_address, port),
            init_temperature=init_temperature,
            reduce_rate=reduce_rate,
            search_steps=args.search_steps,
            is_server=False)

    image_shape = [3, image_size, image_size]
    for step in range(args.search_steps):
        archs = sa_nas.next_archs()[0]

        train_program = fluid.Program()
        test_program = fluid.Program()
        startup_program = fluid.Program()
        train_loader, avg_cost, acc_top1, acc_top5 = build_program(
            train_program, startup_program, image_shape, archs, args)

        current_flops = flops(train_program)
        print('step: {}, current_flops: {}'.format(step, current_flops))
        if current_flops > max_flops:
            continue

        test_loader, test_avg_cost, test_acc_top1, test_acc_top5 = build_program(
            test_program,
            startup_program,
            image_shape,
            archs,
            args,
            is_test=True)
        test_program = test_program.clone(for_test=True)

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
            test_fetches = [
                test_avg_cost.name, test_acc_top1.name, test_acc_top5.name
            ]
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


def test_search_result(tokens, image_size, args, config):
    sa_nas = SANAS(
        config,
        server_addr=("", 8887),
        init_temperature=args.init_temperature,
        reduce_rate=args.reduce_rate,
        search_steps=args.search_steps,
        is_server=True)

    image_shape = [3, image_size, image_size]

    archs = sa_nas.tokens2arch(tokens)

    train_program = fluid.Program()
    test_program = fluid.Program()
    startup_program = fluid.Program()
    train_loader, avg_cost, acc_top1, acc_top5 = build_program(
        train_program, startup_program, image_shape, archs, args)

    current_flops = flops(train_program)
    print('current_flops: {}'.format(current_flops))
    test_loader, test_avg_cost, test_acc_top1, test_acc_top5 = build_program(
        test_program, startup_program, image_shape, archs, args, is_test=True)

    test_program = test_program.clone(for_test=True)

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
            imagenet_reader.val(), batch_size=args.batch_size, drop_last=False)

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
                    'TRAIN: epoch: {}, batch: {}, cost: {}, batch_time: {}ms'.
                    format(epoch_id, batch_id, outs[0], batch_time))

        reward = []
        for batch_id, data in enumerate(test_loader()):
            test_fetches = [
                test_avg_cost.name, test_acc_top1.name, test_acc_top5.name
            ]
            batch_reward = exe.run(test_program,
                                   feed=data,
                                   fetch_list=test_fetches)
            reward_avg = np.mean(np.array(batch_reward), axis=1)
            reward.append(reward_avg)

            _logger.info(
                'TEST: batch: {}, avg_cost: {}, acc_top1: {}, acc_top5: {}'.
                format(batch_id, batch_reward[0], batch_reward[1],
                       batch_reward[2]))

        finally_reward = np.mean(np.array(reward), axis=0)
        _logger.info(
            'FINAL TEST: avg_cost: {}, acc_top1: {}, acc_top5: {}'.format(
                finally_reward[0], finally_reward[1], finally_reward[2]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='SA NAS MobileNetV2 cifar10 argparase')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='Whether to use GPU in train/test model.')
    parser.add_argument(
        '--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument(
        '--data',
        type=str,
        default='cifar10',
        choices=['cifar10', 'imagenet'],
        help='server address.')
    parser.add_argument(
        '--is_server',
        type=ast.literal_eval,
        default=True,
        help='Whether to start a server.')
    parser.add_argument(
        '--search_steps',
        type=int,
        default=100,
        help='controller server number.')
    parser.add_argument(
        '--lr_strategy',
        type=str,
        default='cosine_decay',
        help='learning rate decay strategy.')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
    parser.add_argument(
        '--l2_decay', type=float, default=1e-4, help='learning rate decay.')
    parser.add_argument(
        '--class_dim', type=int, default=100, help='classify number.')
    parser.add_argument(
        '--step_epochs',
        nargs='+',
        type=int,
        default=[30, 60, 90],
        help="piecewise decay step")
    parser.add_argument(
        '--momentum_rate',
        type=float,
        default=0.9,
        help='learning rate decay.')
    parser.add_argument(
        '--warm_up_epochs',
        type=float,
        default=5.0,
        help='learning rate decay.')
    parser.add_argument(
        '--num_epochs', type=int, default=120, help='learning rate decay.')
    parser.add_argument(
        '--decay_epochs', type=float, default=2.4, help='learning rate decay.')
    parser.add_argument(
        '--decay_rate', type=float, default=0.97, help='learning rate decay.')
    parser.add_argument(
        '--total_images',
        type=int,
        default=1281167,
        help='learning rate decay.')
    args = parser.parse_args()
    print(args)

    if args.data == 'cifar10':
        image_size = 32
        block_num = 3
    elif args.data == 'imagenet':
        image_size = 224
        block_num = 6
    else:
        raise NotImplementedError(
            'data must in [cifar10, imagenet], but received: {}'.format(
                args.data))

    config = [('MobileNetV2Space')]

    search_mobilenetv2(config, args, image_size, is_server=args.is_server)

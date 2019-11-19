import sys
sys.path.append('..')
import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.nas.search_space.search_space_factory import SearchSpaceFactory
from paddleslim.analysis import flops
from paddleslim.nas import SANAS


def create_data_loader():
    data = fluid.data(name='data', shape=[-1, 3, 32, 32], dtype='float32')
    label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[data, label],
        capacity=1024,
        use_double_buffer=True,
        iterable=True)
    return data_loader, data, label


def init_sa_nas(config):
    factory = SearchSpaceFactory()
    space = factory.get_search_space(config)
    model_arch = space.token2arch()[0]
    main_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(main_program, startup_program):
        data_loader, data, label = create_data_loader()
        output = model_arch(data)
        output = fluid.layers.fc(
            input=output,
            size=args.class_dim,
            param_attr=ParamAttr(name='mobilenetv2_fc_weights'),
            bias_attr=ParamAttr(name='mobilenetv2_fc_offset'))
        cost = fluid.layers.mean(
            fluid.layers.softmax_with_cross_entropy(
                logits=output, label=label))

        base_flops = flops(main_program)
        search_steps = 10000000

        ### start a server and a client
        sa_nas = SANAS(config, max_flops=base_flops, search_steps=search_steps)

        ### start a client, server_addr is server address
        #sa_nas = SANAS(config, max_flops = base_flops, server_addr=("10.255.125.38", 18607), search_steps = search_steps, is_server=False)

    return sa_nas, search_steps


def search_mobilenetv2_cifar10(config, args):
    sa_nas, search_steps = init_sa_nas(config)
    for i in range(search_steps):
        print('search step: ', i)
        archs = sa_nas.next_archs()[0]

        train_program = fluid.Program()
        test_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            train_loader, data, label = create_data_loader()
            output = archs(data)
            output = fluid.layers.fc(
                input=output,
                size=args.class_dim,
                param_attr=ParamAttr(name='mobilenetv2_fc_weights'),
                bias_attr=ParamAttr(name='mobilenetv2_fc_offset'))
            cost = fluid.layers.mean(
                fluid.layers.softmax_with_cross_entropy(
                    logits=output, label=label))[0]
            test_program = train_program.clone(for_test=True)

            optimizer = fluid.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
            optimizer.minimize(cost)

        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        train_reader = paddle.reader.shuffle(
            paddle.dataset.cifar.train10(cycle=False), buf_size=1024)
        train_loader.set_sample_generator(
            train_reader,
            batch_size=512,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())

        test_loader, _, _ = create_data_loader()
        test_reader = paddle.dataset.cifar.test10(cycle=False)
        test_loader.set_sample_generator(
            test_reader,
            batch_size=256,
            drop_last=False,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())

        for epoch_id in range(10):
            for batch_id, data in enumerate(train_loader()):
                loss = exe.run(train_program,
                               feed=data,
                               fetch_list=[cost.name])[0]
                if batch_id % 5 == 0:
                    print('epoch: {}, batch: {}, loss: {}'.format(
                        epoch_id, batch_id, loss[0]))

        for data in test_loader():
            reward = exe.run(test_program, feed=data,
                             fetch_list=[cost.name])[0]

        print('reward:', reward)
        sa_nas.reward(float(reward))


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
    args = parser.parse_args()
    print(args)

    # block mask means block number, 1 mean downsample, 0 means the size of feature map don't change after this block
    config_info = {
        'input_size': 32,
        'output_size': 1,
        'block_num': 5,
        'block_mask': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    }
    config = [('MobileNetV2BlockSpace', config_info)]

    search_mobilenetv2_cifar10(config, args)

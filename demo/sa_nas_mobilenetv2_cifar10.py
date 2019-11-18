import sys
sys.path.append('..')
import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddleslim.nas.search_space.search_space_factory import SearchSpaceFactory
from paddleslim.analysis import flops
from paddleslim.nas import SANAS


def create_data_loader():
    data = fluid.data(name='data', shape=[-1, 3, 32, 32], dtype='float32')
    label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[data, label],
        capacity=64,
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
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(cycle=False), buf_size=1024),
            batch_size=512)
        train_loader.set_batch_generator(
            train_reader,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())

        test_loader, _, _ = create_data_loader()
        test_reader = paddle.batch(
            paddle.dataset.cifar.test10(cycle=False), batch_size=256)
        test_loader.set_batch_generator(
            test_reader,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())

        data_loader.set_batch_generator

        for epoch_id in range(10):
            for batch_id, data in enumerate(train_loader()):
                real_image = np.array(list(map(lambda x: x[0], data))).reshape(
                    -1, 3, 32, 32).astype('float32')
                real_label = np.array(list(map(lambda x: x[1], data))).reshape(
                    -1, 1).astype('int64')
                loss = exe.run(train_program,
                               feed={'data': real_image,
                                     'label': real_label},
                               fetch_list=[cost.name])[0]
                if batch_id % 5 == 0:
                    print('epoch: {}, batch: {}, loss: {}'.format(
                        epoch_id, batch_id, loss[0]))

        for data in test_loader():
            test_image = np.array(list(map(lambda x: x[0], data))).reshape(
                -1, 3, 32, 32).astype('float32')
            test_label = np.array(list(map(lambda x: x[1], data))).reshape(
                -1, 1).astype('int64')
            reward = exe.run(test_program,
                             feed={'data': test_image,
                                   'label': test_label},
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
    args = parser.parse_args()
    print(args)

    config_info = {'input_size': 32, 'output_size': 1, 'block_num': 5}
    config = [('MobileNetV2Space', config_info)]

    search_mobilenetv2_cifar10(config, args)

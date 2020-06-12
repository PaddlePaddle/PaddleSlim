import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
import paddle.fluid as fluid
sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.analysis import flops
from paddleslim.quant import quant_aware, quant_post, convert
import models
from utility import add_arguments, print_arguments
from pact import *
quantization_model_save_dir = './quantization_models/'

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64 * 4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  "../pretrained_model/MobileNetV1_pretrained",                "Whether to use pretrained model.")
add_arg('lr',               float,  0.0001,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  1,               "The number of total epochs.")
add_arg('total_images',     int,  1281167,               "The number of total training images.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('config_file',      str, None,                 "The config file for compression with yaml format.")
add_arg('data',             str, "imagenet",             "Which data to use. 'mnist' or 'imagenet'")
add_arg('log_period',       int, 10,                 "Log period in batches.")
add_arg('checkpoint_dir',         str, "output",           "checkpoint save dir")
add_arg('use_pact',          bool, True,                "Whether to use PACT or not.")

# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def piecewise_decay(args):
    step = int(math.ceil(float(args.total_images) / args.batch_size))
    bd = [step * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay))
    return optimizer


def cosine_decay(args):
    step = int(math.ceil(float(args.total_images) / args.batch_size))
    learning_rate = fluid.layers.cosine_decay(
        learning_rate=args.lr, step_each_epoch=step, epochs=args.num_epochs)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay))
    return optimizer


def create_optimizer(args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args)


def compress(args):
    # 1. quantization configs
    quant_config = {
        # weight quantize type, default is 'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',
        # activation quantize type, default is 'moving_average_abs_max'
        'activation_quantize_type': 'moving_average_abs_max',
        # weight quantize bit num, default is 8
        'weight_bits': 8,
        # activation quantize bit num, default is 8
        'activation_bits': 8,
        # ops of name_scope in not_quant_pattern list, will not be quantized
        'not_quant_pattern': ['skip_quant'],
        # ops of type in quantize_op_types, will be quantized
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
        # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
        'dtype': 'int8',
        # window size for 'range_abs_max' quantization. defaulf is 10000
        'window_size': 10000,
        # The decay coefficient of moving average, default is 0.9
        'moving_rate': 0.9,
    }

    train_reader = None
    test_reader = None
    if args.data == "mnist":
        import paddle.dataset.mnist as reader
        train_reader = reader.train()
        val_reader = reader.test()
        class_dim = 10
        image_shape = "1,28,28"
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_reader = reader.train()
        val_reader = reader.val()
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))

    image_shape = [int(m) for m in image_shape.split(",")]
    assert args.model in model_list, "{} is not in lists: {}".format(
        args.model, model_list)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    if args.use_pact:
        image.stop_gradient = False
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    train_prog = fluid.default_main_program()
    val_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    opt = create_optimizer(args)
    opt.minimize(avg_cost)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 2. quantization transform programs (training aware)
    #    Make some quantization transforms in the graph before training and testing.
    #    According to the weight and activation quantization type, the graph will be added
    #    some fake quantize operators and fake dequantize operators.

    if args.use_pact:
        act_preprocess_func = pact
        optimizer_func = get_optimizer
        executor = exe
    else:
        act_preprocess_func = None
        optimizer_func = None
        executor = None

    val_program = quant_aware(
        val_program,
        place,
        quant_config,
        scope=None,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=executor,
        for_test=True)
    compiled_train_prog = quant_aware(
        train_prog,
        place,
        quant_config,
        scope=None,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=executor,
        for_test=False)

    assert os.path.exists(
        args.pretrained_model), "pretrained_model doesn't exist"

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(
                os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    val_reader = paddle.fluid.io.batch(val_reader, batch_size=args.batch_size)
    train_reader = paddle.fluid.io.batch(
        train_reader, batch_size=args.batch_size, drop_last=True)

    train_feeder = feeder = fluid.DataFeeder([image, label], place)
    val_feeder = feeder = fluid.DataFeeder(
        [image, label], place, program=val_program)

    def test(epoch, program):
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []
        for data in val_reader():
            start_time = time.time()
            acc_top1_n, acc_top5_n = exe.run(
                program,
                feed=train_feeder.feed(data),
                fetch_list=[acc_top1.name, acc_top5.name])
            end_time = time.time()
            if batch_id % args.log_period == 0:
                _logger.info(
                    "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id,
                           np.mean(acc_top1_n),
                           np.mean(acc_top5_n), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1_n))
            acc_top5_ns.append(np.mean(acc_top5_n))
            batch_id += 1

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".
                     format(epoch,
                            np.mean(np.array(acc_top1_ns)),
                            np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))

    def train(epoch, compiled_train_prog):

        batch_id = 0
        for data in train_reader():
            start_time = time.time()
            loss_n, acc_top1_n, acc_top5_n = exe.run(
                compiled_train_prog,
                feed=train_feeder.feed(data),
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            end_time = time.time()
            loss_n = np.mean(loss_n)
            acc_top1_n = np.mean(acc_top1_n)
            acc_top5_n = np.mean(acc_top5_n)
            if batch_id % args.log_period == 0:
                _logger.info(
                    "epoch[{}]-batch[{}] - loss: {}; acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id, loss_n, acc_top1_n, acc_top5_n,
                           end_time - start_time))

            if args.use_pact and batch_id % 1000 == 0:
                threshold = {}
                for var in val_program.list_vars():
                    if 'pact' in var.name:
                        array = np.array(fluid.global_scope().find_var(
                            var.name).get_tensor())
                        threshold[var.name] = array[0]
                print(threshold)

            batch_id += 1

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.sync_batch_norm = False
    exec_strategy = fluid.ExecutionStrategy()
    compiled_train_prog = compiled_train_prog.with_data_parallel(
        loss_name=avg_cost.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    ############################################################################################################
    # train loop
    ############################################################################################################
    best_acc1 = 0.0
    best_epoch = 0
    for i in range(args.num_epochs):
        train(i, compiled_train_prog)
        acc1 = test(i, val_program)
        fluid.io.save_persistables(
            exe,
            dirname=os.path.join(args.checkpoint_dir, str(i)),
            main_program=val_program)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = i
            fluid.io.save_persistables(
                exe,
                dirname=os.path.join(args.checkpoint_dir, 'best_model'),
                main_program=val_program)
    if os.path.exists(os.path.join(args.checkpoint_dir, 'best_model')):
        fluid.io.load_persistables(
            exe,
            dirname=os.path.join(args.checkpoint_dir, 'best_model'),
            main_program=val_program)
    ############################################################################################################
    # 3. Freeze the graph after training by adjusting the quantize
    #    operators' order for the inference.
    #    The dtype of float_program's weights is float32, but in int8 range.
    ############################################################################################################
    float_program, int8_program = convert(val_program, place, quant_config, \
                                                        scope=None, \
                                                        save_int8=True)
    print("eval best_model after convert")
    final_acc1 = test(best_epoch, float_program)
    ############################################################################################################
    # 4. Save inference model
    ############################################################################################################
    model_path = os.path.join(quantization_model_save_dir, args.model,
                              'act_' + quant_config['activation_quantize_type']
                              + '_w_' + quant_config['weight_quantize_type'])
    float_path = os.path.join(model_path, 'float')
    int8_path = os.path.join(model_path, 'int8')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    fluid.io.save_inference_model(
        dirname=float_path,
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe,
        main_program=float_program,
        model_filename=float_path + '/model',
        params_filename=float_path + '/params')

    fluid.io.save_inference_model(
        dirname=int8_path,
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe,
        main_program=int8_program,
        model_filename=int8_path + '/model',
        params_filename=int8_path + '/params')


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

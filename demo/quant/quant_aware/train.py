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
from paddleslim.quant import quant_aware, convert
import models
from utility import add_arguments, print_arguments

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
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def piecewise_decay(args):
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    step = int(
        math.ceil(float(args.total_images) / (args.batch_size * len(places))))
    bd = [step * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer


def cosine_decay(args):
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    step = int(
        math.ceil(float(args.total_images) / (args.batch_size * len(places))))
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=step * args.num_epochs, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer


def create_optimizer(args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args)


def compress(args):
    ############################################################################################################
    # 1. quantization configs
    ############################################################################################################
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

    if args.data == "mnist":
        train_dataset = paddle.vision.datasets.MNIST(mode='train')
        val_dataset = paddle.vision.datasets.MNIST(mode='test')
        class_dim = 10
        image_shape = "1,28,28"
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))

    image_shape = [int(m) for m in image_shape.split(",")]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    train_prog = paddle.static.default_main_program()
    val_program = paddle.static.default_main_program().clone(for_test=True)

    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    ############################################################################################################
    # 2. quantization transform programs (training aware)
    #    Make some quantization transforms in the graph before training and testing.
    #    According to the weight and activation quantization type, the graph will be added
    #    some fake quantize operators and fake dequantize operators.
    ############################################################################################################
    val_program = quant_aware(
        val_program, place, quant_config, scope=None, for_test=True)
    compiled_train_prog = quant_aware(
        train_prog, place, quant_config, scope=None, for_test=False)
    opt = create_optimizer(args)
    opt.minimize(avg_cost)

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    assert os.path.exists(
        args.pretrained_model), "pretrained_model doesn't exist"

    if args.pretrained_model:
        paddle.static.load(train_prog, args.pretrained_model, exe)

    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()

    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=places,
        feed_list=[image, label],
        drop_last=True,
        batch_size=args.batch_size,
        return_list=False,
        use_shared_memory=True,
        shuffle=True,
        num_workers=4)
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        batch_size=args.batch_size,
        use_shared_memory=True,
        shuffle=False)

    def test(epoch, program):
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []
        for data in valid_loader():
            start_time = time.time()
            acc_top1_n, acc_top5_n = exe.run(
                program, feed=data, fetch_list=[acc_top1.name, acc_top5.name])
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

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))

    def train(epoch, compiled_train_prog):

        batch_id = 0
        for data in train_loader():
            start_time = time.time()
            loss_n, acc_top1_n, acc_top5_n = exe.run(
                compiled_train_prog,
                feed=data,
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
            batch_id += 1

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.sync_batch_norm = False
    exec_strategy = paddle.static.ExecutionStrategy()
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
        paddle.static.save(
            program=val_program,
            model_path=os.path.join(args.checkpoint_dir, str(i)))
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = i
            paddle.static.save(
                program=val_program,
                model_path=os.path.join(args.checkpoint_dir, 'best_model'))
    if os.path.exists(os.path.join(args.checkpoint_dir, 'best_model')):
        paddle.static.load(
            executor=exe,
            model_path=os.path.join(args.checkpoint_dir, 'best_model'),
            program=val_program)
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
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    paddle.fluid.io.save_inference_model(
        dirname=float_path,
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe,
        main_program=float_program,
        model_filename=float_path + '/model',
        params_filename=float_path + '/params')


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

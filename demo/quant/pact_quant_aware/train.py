import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname("__file__"))
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
from paddleslim.common import get_logger, VarCollector
from paddleslim.analysis import flops
from paddleslim.quant import quant_aware, quant_post, convert
import models
from utility import add_arguments, print_arguments
from paddle.fluid.layer_helper import LayerHelper
quantization_model_save_dir = './quantization_models/'

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  128,
        "Minibatch size.")
add_arg('use_gpu',          bool, True,
        "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNetV3_large_x1_0",
        "The target model.")
add_arg('pretrained_model', str,  "./pretrain/MobileNetV3_large_x1_0_ssld_pretrained",
        "Whether to use pretrained model.")
add_arg('lr',               float,  0.001,
        "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",
        "The learning rate decay strategy.")
add_arg('l2_decay',         float,  1e-5,
        "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,
        "The value of momentum_rate.")
add_arg('num_epochs',       int,  30,
        "The number of total epochs.")
add_arg('total_images',     int,  1281167,
        "The number of total training images.")
parser.add_argument('--step_epochs', nargs='+', type=int,
        default=[20],
        help="piecewise decay step")
add_arg('config_file',      str, None,
        "The config file for compression with yaml format.")
add_arg('data',             str, "imagenet",
        "Which data to use. 'mnist' or 'imagenet'")
add_arg('log_period',       int, 10,
        "Log period in batches.")
add_arg('checkpoint_dir',         str, None,
        "checkpoint dir")
add_arg('checkpoint_epoch',         int, None,
        "checkpoint epoch")
add_arg('output_dir',         str, "output/MobileNetV3_large_x1_0",
        "model save dir")
add_arg('use_pact',          bool, True,
        "Whether to use PACT or not.")
add_arg('analysis',          bool, False,
        "Whether analysis variables distribution.")

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
    return learning_rate, optimizer


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
    return learning_rate, optimizer


def create_optimizer(args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args)


def compress(args):

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
    if args.use_pact:
        image.stop_gradient = False
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

    if not args.analysis:
        learning_rate, opt = create_optimizer(args)
        opt.minimize(avg_cost)

    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=places,
        feed_list=[image, label],
        drop_last=True,
        return_list=False,
        batch_size=args.batch_size,
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

    if args.analysis:
        # get all activations names
        activates = [
            'pool2d_1.tmp_0', 'tmp_35', 'batch_norm_21.tmp_2', 'tmp_26',
            'elementwise_mul_5.tmp_0', 'pool2d_5.tmp_0',
            'elementwise_add_5.tmp_0', 'relu_2.tmp_0', 'pool2d_3.tmp_0',
            'conv2d_40.tmp_2', 'elementwise_mul_0.tmp_0', 'tmp_62',
            'elementwise_add_8.tmp_0', 'batch_norm_39.tmp_2', 'conv2d_32.tmp_2',
            'tmp_17', 'tmp_5', 'elementwise_add_9.tmp_0', 'pool2d_4.tmp_0',
            'relu_0.tmp_0', 'tmp_53', 'relu_3.tmp_0', 'elementwise_add_4.tmp_0',
            'elementwise_add_6.tmp_0', 'tmp_11', 'conv2d_36.tmp_2',
            'relu_8.tmp_0', 'relu_5.tmp_0', 'pool2d_7.tmp_0',
            'elementwise_add_2.tmp_0', 'elementwise_add_7.tmp_0',
            'pool2d_2.tmp_0', 'tmp_47', 'batch_norm_12.tmp_2',
            'elementwise_mul_6.tmp_0', 'elementwise_mul_7.tmp_0',
            'pool2d_6.tmp_0', 'relu_6.tmp_0', 'elementwise_add_0.tmp_0',
            'elementwise_mul_3.tmp_0', 'conv2d_12.tmp_2',
            'elementwise_mul_2.tmp_0', 'tmp_8', 'tmp_2', 'conv2d_8.tmp_2',
            'elementwise_add_3.tmp_0', 'elementwise_mul_1.tmp_0',
            'pool2d_8.tmp_0', 'conv2d_28.tmp_2', 'image', 'conv2d_16.tmp_2',
            'batch_norm_33.tmp_2', 'relu_1.tmp_0', 'pool2d_0.tmp_0', 'tmp_20',
            'conv2d_44.tmp_2', 'relu_10.tmp_0', 'tmp_41', 'relu_4.tmp_0',
            'elementwise_add_1.tmp_0', 'tmp_23', 'batch_norm_6.tmp_2', 'tmp_29',
            'elementwise_mul_4.tmp_0', 'tmp_14'
        ]
        var_collector = VarCollector(train_prog, activates, use_ema=True)
        values = var_collector.abs_max_run(
            train_loader, exe, step=None, loss_name=avg_cost.name)
        np.save('pact_thres.npy', values)
        _logger.info(values)
        _logger.info("PACT threshold have been saved as pact_thres.npy")

        # Draw Histogram in 'dist_pdf/result.pdf'
        # var_collector.pdf(values)

        return

    values = defaultdict(lambda: 20)
    try:
        values = np.load("pact_thres.npy", allow_pickle=True).item()
        values.update(tmp)
        _logger.info("pact_thres.npy info loaded.")
    except:
        _logger.info(
            "cannot find pact_thres.npy. Set init PACT threshold as 20.")
    _logger.info(values)

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

    # 2. quantization transform programs (training aware)
    #    Make some quantization transforms in the graph before training and testing.
    #    According to the weight and activation quantization type, the graph will be added
    #    some fake quantize operators and fake dequantize operators.

    def pact(x):
        helper = LayerHelper("pact", **locals())
        dtype = 'float32'
        init_thres = values[x.name.split('_tmp_input')[0]]
        u_param_attr = paddle.ParamAttr(
            name=x.name + '_pact',
            initializer=paddle.nn.initializer.Constant(value=init_thres),
            regularizer=paddle.regularizer.L2Decay(0.0001),
            learning_rate=1)
        u_param = helper.create_parameter(
            attr=u_param_attr, shape=[1], dtype=dtype)

        part_a = paddle.nn.functional.relu(x - u_param)
        part_b = paddle.nn.functional.relu(-u_param - x)
        x = x - part_a + part_b
        return x

    def get_optimizer():
        return paddle.optimizer.Momentum(args.lr, 0.9)

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
        paddle.static.load(train_prog, args.pretrained_model, exe)

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
                    "Eval epoch[{}] batch[{}] - acc_top1: {:.6f}; acc_top5: {:.6f}; time: {:.3f}".
                    format(epoch, batch_id,
                           np.mean(acc_top1_n),
                           np.mean(acc_top5_n), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1_n))
            acc_top5_ns.append(np.mean(acc_top5_n))
            batch_id += 1

        _logger.info(
            "Final eval epoch[{}] - acc_top1: {:.6f}; acc_top5: {:.6f}".format(
                epoch,
                np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))

    def train(epoch, compiled_train_prog, lr):

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
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {:.6f}; acc_top1: {:.6f}; acc_top5: {:.6f}; time: {:.3f}".
                    format(epoch, batch_id,
                           learning_rate.get_lr(), loss_n, acc_top1_n,
                           acc_top5_n, end_time - start_time))

            if args.use_pact and batch_id % 1000 == 0:
                threshold = {}
                for var in val_program.list_vars():
                    if 'pact' in var.name:
                        array = np.array(paddle.static.global_scope().find_var(
                            var.name).get_tensor())
                        threshold[var.name] = array[0]
                _logger.info(threshold)
            batch_id += 1
            lr.step()

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_reduce_ops = False
    exec_strategy = paddle.static.ExecutionStrategy()
    compiled_train_prog = compiled_train_prog.with_data_parallel(
        loss_name=avg_cost.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    # train loop
    best_acc1 = 0.0
    best_epoch = 0

    start_epoch = 0
    if args.checkpoint_dir is not None:
        ckpt_path = args.checkpoint_dir
        assert args.checkpoint_epoch is not None, "checkpoint_epoch must be set"
        start_epoch = args.checkpoint_epoch
        paddle.static.load(
            executor=exe, model_path=args.checkpoint_dir, program=val_program)

    best_eval_acc1 = 0
    best_acc1_epoch = 0
    for i in range(start_epoch, args.num_epochs):
        train(i, compiled_train_prog, learning_rate)
        acc1 = test(i, val_program)
        if acc1 > best_eval_acc1:
            best_eval_acc1 = acc1
            best_acc1_epoch = i
        _logger.info("Best Validation Acc1: {:.6f}, at epoch {}".format(
            best_eval_acc1, best_acc1_epoch))
        paddle.static.save(
            model_path=os.path.join(args.output_dir, str(i)),
            program=val_program)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = i
            paddle.static.save(
                model_path=os.path.join(args.output_dir, 'best_model'),
                program=val_program)

    if os.path.exists(os.path.join(args.output_dir, 'best_model.pdparams')):
        paddle.static.load(
            executor=exe,
            model_path=os.path.join(args.output_dir, 'best_model'),
            program=val_program)

    # 3. Freeze the graph after training by adjusting the quantize
    #    operators' order for the inference.
    #    The dtype of float_program's weights is float32, but in int8 range.
    float_program, int8_program = convert(val_program, place, quant_config, \
                                                        scope=None, \
                                                        save_int8=True)
    _logger.info("eval best_model after convert")
    final_acc1 = test(best_epoch, float_program)
    _logger.info("final acc:{}".format(final_acc1))

    # 4. Save inference model
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

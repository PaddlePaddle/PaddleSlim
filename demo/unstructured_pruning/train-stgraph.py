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
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname("__file__")))
from unstructure_pruner_stgraph import UnstructurePruner
from paddleslim.prune import Pruner, save_model
import paddle.distributed as dist
from paddleslim.common import get_logger
from paddleslim.analysis import flops
import models
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64*8,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  "../pretrained_model/MobileNetV1_pretrained",                "Whether to use pretrained model.")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('threshold_value',        float,  1e-5,               "The threshold to set zeros, the abs(weights) lower than which will be zeros.")
add_arg('pruning_mode',            str,  'ratio',               "the pruning mode: whether by ratio or by threshold.")
add_arg('ratio_value',        float,  0.5,               "The threshold to set zeros, the abs(weights) lower than which will be zeros.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'.")
add_arg('log_period',       int, 100,                 "Log period in batches.")
add_arg('phase',            str, "train",              "Whether to train or test the pruned model.")
add_arg('test_period',      int, 10,                 "Test period in epoches.")
add_arg('model_path',       str, "./models",         "The path to save model.")
add_arg('model_period',     int, 10,             "The period to save model in epochs.")
add_arg('resume_epoch',     int, 0,             "The epoch to resume training.")
# yapf: enable

model_list = models.__all__


def get_pruned_params(args, program):
    params = []
    if args.model == "MobileNet":
        for param in program.global_block().all_parameters():
            if "_sep_weights" in param.name:
                params.append(param.name)
    elif args.model == "MobileNetV2":
        for param in program.global_block().all_parameters():
            if "linear_weights" in param.name or "expand_weights" in param.name:
                params.append(param.name)
    elif args.model == "ResNet34":
        for param in program.global_block().all_parameters():
            if "weights" in param.name and "branch" in param.name:
                params.append(param.name)
    elif args.model == "PVANet":
        for param in program.global_block().all_parameters():
            if "conv_weights" in param.name:
                params.append(param.name)
    return params


def piecewise_decay(args, step_per_epoch):
    bd = [step_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def cosine_decay(args, step_per_epoch):
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=args.num_epochs * step_per_epoch)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def create_optimizer(args, step_per_epoch):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args, step_per_epoch)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args, step_per_epoch)


def compress(args):
    train_reader = None
    test_reader = None
    if args.data == "mnist":
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend="cv2", transform=transform)
        val_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend="cv2", transform=transform)
        class_dim = 10
        image_shape = "1,28,28"
        args.pretrained_model = False
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(data_dir='/data', mode='train')
        val_dataset = reader.ImageNetDataset(data_dir='/data', mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))
    image_shape = [int(m) for m in image_shape.split(",")]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    place = places[0]
    exe = paddle.static.Executor(place)
    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    threshold = paddle.static.data(
        name='pruning_threshold', shape=[None, 1], dtype='float32')

    batch_size_per_card = int(args.batch_size / len(places))
    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=places,
        feed_list=[image, label],
        drop_last=True,
        batch_size=batch_size_per_card,
        shuffle=True,
        return_list=False,
        use_shared_memory=True,
        num_workers=32)
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        use_shared_memory=True,
        batch_size=batch_size_per_card,
        shuffle=False)
    step_per_epoch = int(np.ceil(len(train_dataset) * 1. / args.batch_size))

    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    val_program = paddle.static.default_main_program().clone(for_test=True)

    opt, learning_rate = create_optimizer(args, step_per_epoch)
    opt.minimize(avg_cost)

    pruner = UnstructurePruner(
        paddle.static.default_main_program(),
        batch_size=args.batch_size,
        mode=args.pruning_mode,
        threshold=threshold,
        ratio_value=args.ratio_value,
        place=place)

    exe.run(paddle.static.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        _logger.info("Load pretrained model from {}".format(
            args.pretrained_model))
        paddle.fluid.io.load_vars(
            exe, args.pretrained_model, predicate=if_exist)

    def test(epoch, program):
        acc_top1_ns = []
        acc_top5_ns = []

        print("The current density of the inference model is {}%".format(
            round(100 * UnstructurePruner.total_sparse(
                paddle.static.default_main_program()), 2)))
        for batch_id, data in enumerate(valid_loader):
            start_time = time.time()
            acc_top1_n, acc_top5_n = exe.run(
                program,
                feed={
                    "image": data[0].get('image'),
                    "label": data[0].get('label'),
                    "pruning_threshold": pruner.threshold_values
                },
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

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))

    def train(epoch, program):
        for batch_id, data in enumerate(train_loader):
            start_time = time.time()
            loss_n, acc_top1_n, acc_top5_n = exe.run(
                train_program,
                feed={
                    "image": data[0].get('image'),
                    "label": data[0].get('label'),
                    "pruning_threshold": pruner.threshold_values
                },
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            end_time = time.time()
            loss_n = np.mean(loss_n)
            acc_top1_n = np.mean(acc_top1_n)
            acc_top5_n = np.mean(acc_top5_n)
            if batch_id % args.log_period == 0:
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {}; acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id,
                           learning_rate.get_lr(), loss_n, acc_top1_n,
                           acc_top5_n, end_time - start_time))
            learning_rate.step()
            pruner.step()
            batch_id += 1

    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    if args.phase == 'train':
        train_program = paddle.static.CompiledProgram(
            paddle.static.default_main_program()).with_data_parallel(
                loss_name=avg_cost.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        for i in range(args.resume_epoch, args.num_epochs):
            train(i, train_program)
            print("The current density of the pruned model is: {}%".format(
                round(100 * UnstructurePruner.total_sparse(
                    paddle.static.default_main_program()), 2)))

            if i % args.test_period == 0:
                pruner.update_params()
                test(i, val_program)
            if i > args.resume_epoch and i % args.model_period == 0:
                pruner.update_params()
                # fluid.io.save_params(executor=exe, dirname=args.model_path)
    elif args.phase == 'test':
        test(0, val_program)


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

import os
import sys
import logging
import paddle
import argparse
import functools
import time
import numpy as np
import paddle.fluid as fluid
from paddleslim.prune.unstructured_pruner import UnstructuredPruner
from paddleslim.common import get_logger
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
import models
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
import cifar
from paddleslim.core import GraphWrapper

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  128,                 "Minibatch size.")
add_arg('batch_size_for_validation',       int,  64,                 "Minibatch size for validation.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNetCifar",                "The target model.")
add_arg('pretrained_model', str,  "/code/models/pretrained_cifar100",                "Whether to use pretrained model.")
add_arg('lr',               float,  0.01,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  5e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('threshold',        float,  5e-5,               "The threshold to set zeros, the abs(weights) lower than which will be zeros.")
add_arg('pruning_mode',            str,  'ratio',               "the pruning mode: whether by ratio or by threshold.")
add_arg('ratio',        float,  0.90,               "The ratio to set zeros, the smaller portion will be zeros.")
add_arg('num_epochs',       int,  300,               "The number of total epochs.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[220, 260], help="piecewise decay step")
add_arg('data',             str, "cifar100",                 "Which data to use. 'mnist' or 'imagenet'.")
add_arg('log_period',       int, 10000,                 "Log period in batches.")
add_arg('test_period',      int, 1,                 "Test period in epoches.")
add_arg('model_path',       str, "./models",         "The path to save model.")
add_arg('model_period',     int, 10,             "The period to save model in epochs.")
add_arg('resume_epoch',     int, -1,             "The epoch to resume training.")
add_arg('stable_epochs',    int, 2,              "The epoch numbers used to stablize the model before pruning. Default: 2")
add_arg('pruning_epochs',   int, 180,             "The epoch numbers used to prune the model by a ratio step. Default: 35")
add_arg('tunning_epochs',   int, 120,             "The epoch numbers used to tune the after-pruned models. Default: 20")
add_arg('ratio_steps_per_epoch', int, 3,        "How many times you want to increase your ratio during each epoch. Default: 30")
add_arg('initial_ratio',    float, 0.10,         "The initial pruning ratio used at the start of pruning stage. Default: 0.05")
# yapf: enable

model_list = models.__all__


def piecewise_decay(args, step_per_epoch):
    bd = [step_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.2**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def get_skip_params(program):
    skip_params = set()
    graph = GraphWrapper(program)
    for op in graph.ops():
        if 'norm' in op.type() and 'grad' not in op.type():
            for input in op.all_inputs():
                skip_params.add(input.name())

    for param in program.all_parameters():
        cond = len(param.shape) == 4 and param.shape[2] == 1 and param.shape[
            3] == 1
        if not cond: skip_params.add(param.name)

    return skip_params


class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


def cosine_decay(args, step_per_epoch):
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=args.num_epochs * step_per_epoch)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def linear_decay(args, step_per_epoch):
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.lr,
        decay_steps=args.tunning_epochs * step_per_epoch,
        verbose=False)
    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=lr_scheduler,
        warmup_steps=args.stable_epochs + args.pruning_epochs,
        start_lr=args.lr - 1e-5,
        end_lr=args.lr,
        verbose=False)
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
    elif args.lr_strategy == "linear_decay":
        return linear_decay(args, step_per_epoch)


def prepare_training_hyper_parameters_y(args, step_per_epoch):
    total_pruning_steps = args.ratio_steps_per_epoch * args.pruning_epochs
    ratios = []
    ratio_increment_period = int(step_per_epoch / args.ratio_steps_per_epoch)
    for i in range(total_pruning_steps):
        ratio_tmp = ((i / total_pruning_steps) - 1)**3 + 1
        ratio_tmp = ratio_tmp * (args.ratio - args.initial_ratio
                                 ) + args.initial_ratio
        ratios.append(ratio_tmp)
    ratios.reverse()

    return ratios, ratio_increment_period


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
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    elif args.data == "cifar100":
        normalize = T.Normalize(
            mean=[0.5071, 0.4865, 0.4409], std=[0.1942, 0.1918, 0.1958])
        train_transforms = T.Compose([
            T.RandomCrop(
                32, padding=4), T.ContrastTransform(0.1),
            T.BrightnessTransform(0.1), T.RandomHorizontalFlip(),
            T.RandomRotation(15), ToArray(), normalize
        ])
        test_transforms = T.Compose([ToArray(), normalize])
        train_dataset = cifar.Cifar100(
            mode='train', backend='pil', transform=train_transforms)
        val_dataset = cifar.Cifar100(
            mode='test', backend='pil', transform=test_transforms)
        class_dim = 100
        image_shape = "3,32,32"
    else:
        raise ValueError("{} is not supported.".format(args.data))
    image_shape = [int(m) for m in image_shape.split(",")]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    if args.use_gpu:
        places = paddle.static.cuda_places()
    else:
        places = paddle.static.cpu_places()

    place = places[0]
    exe = paddle.static.Executor(place)
    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
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
        num_workers=4)
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        use_shared_memory=True,
        batch_size=args.batch_size_for_validation,
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

    pruner = UnstructuredPruner(
        paddle.static.default_main_program(),
        mode=args.pruning_mode,
        ratio=0.0,
        threshold=0.0,
        place=place,
        skip_params_func=get_skip_params)

    ratios_stack, ratio_increment_period = prepare_training_hyper_parameters_y(
        args, step_per_epoch)

    exe.run(paddle.static.default_startup_program())

    if args.pretrained_model:
        assert os.path.exists(
            args.
            pretrained_model), "Pretrained model path {} doesn't exist".format(
                args.pretrained_model)

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        _logger.info("Load pretrained model from {}".format(
            args.pretrained_model))
        # NOTE: We are using fluid.io.load_vars() because the pretrained model is from an older version which requires this API. 
        #       Please consider using paddle.static.load(program, model_path) when possible
        paddle.fluid.io.load_vars(
            exe, args.pretrained_model, predicate=if_exist)

    def test(epoch, program):
        acc_top1_ns = []
        acc_top5_ns = []

        # _logger.info("The current density of the inference model is {}%".format(
        #     round(100 * UnstructuredPruner.total_sparse_conv1x1(
        #         paddle.static.default_main_program()), 2)))
        for batch_id, data in enumerate(valid_loader):
            start_time = time.time()
            acc_top1_n, acc_top5_n = exe.run(
                program, feed=data, fetch_list=[acc_top1.name, acc_top5.name])
            end_time = time.time()
            if batch_id % args.log_period == 0:
                pass
                # _logger.info(
                #     "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                #     format(epoch, batch_id,
                #            np.mean(acc_top1_n),
                #            np.mean(acc_top5_n), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1_n))
            acc_top5_ns.append(np.mean(acc_top5_n))

        _logger.info(
            "Final eval epoch[{}] ratio: {} - acc_top1: {}; acc_top5: {}".
            format(epoch, pruner.ratio,
                   np.mean(np.array(acc_top1_ns)),
                   np.mean(np.array(acc_top5_ns))))

    def train(epoch, program):
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for batch_id, data in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            loss_n, acc_top1_n, acc_top5_n = exe.run(
                train_program,
                feed=data,
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            ori_ratio = pruner.ratio
            if len(
                    ratios_stack
            ) > 0 and epoch >= args.stable_epochs and epoch < args.stable_epochs + args.pruning_epochs:
                if (batch_id + 1) % ratio_increment_period == 0:
                    pruner.ratio = ratios_stack.pop()
            elif len(
                    ratios_stack
            ) == 0 or epoch >= args.stable_epochs + args.pruning_epochs:
                pruner.ratio = args.ratio

            if ori_ratio != pruner.ratio and epoch >= args.stable_epochs:
                pruner.step()
            train_run_cost += time.time() - train_start
            total_samples += args.batch_size
            loss_n = np.mean(loss_n)
            acc_top1_n = np.mean(acc_top1_n)
            acc_top5_n = np.mean(acc_top5_n)
            '''
            if batch_id % args.log_period == 0:   
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f}; ratio: {:.6f} - loss: {}; acc_top1: {}; acc_top5: {}; avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch, batch_id,
                           learning_rate.get_lr(), pruner.ratio, loss_n,
                           acc_top1_n, acc_top5_n, train_reader_cost /
                           args.log_period, (train_reader_cost + train_run_cost
                                             ) / args.log_period, total_samples
                           / args.log_period, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            '''
            learning_rate.step()
            reader_start = time.time()

    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    train_program = paddle.static.CompiledProgram(
        paddle.static.default_main_program()).with_data_parallel(
            loss_name=avg_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    for i in range(args.resume_epoch + 1, args.num_epochs):
        train(i, train_program)
        # _logger.info("The current density of the pruned model is: {}%".format(
        #     round(100 * UnstructuredPruner.total_sparse_conv1x1(
        #         paddle.static.default_main_program()), 2)))

        if (i + 1) % args.test_period == 0:
            test(i, val_program)
        if (i + 1) % args.model_period == 0:
            # NOTE: We are using fluid.io.save_params() because the pretrained model is from an older version which requires this API. 
            #       Please consider using paddle.static.save(program, model_path) as long as it becomes possible.
            fluid.io.save_params(executor=exe, dirname=args.model_path)


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

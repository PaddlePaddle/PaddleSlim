import os
import sys
import logging
import paddle
import argparse
import functools
import time
import random
import numpy as np
from paddleslim.prune.unstructured_pruner import UnstructuredPruner, GMPUnstructuredPruner
from paddleslim.common import get_logger
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
import models
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
from paddle.distributed import fleet
from paddle.distributed.fleet import DistributedStrategy

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,               "Whether to use gpu for traning or not. Defauly: True")
add_arg('batch_size',       int,  64,                 "Minibatch size. Default: 64")
add_arg('batch_size_for_validation',       int,  64,                 "Minibatch size for validation. Default: 64")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model. Default: None")
add_arg('checkpoint',       str, None, "The model to load for resuming training. Default: None")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model. Default: 0.1")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy. Default: piecewise_decay")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter. Default: 3e-5")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate. Default: 0.9")
add_arg('pruning_strategy', str,    'base',            "The pruning strategy, currently we support base and gmp. Default: base")
add_arg('threshold',        float,  0.01,               "The threshold to set zeros, the abs(weights) lower than which will be zeros. Default: 0.01")
add_arg('pruning_mode',            str,  'ratio',               "the pruning mode: whether by ratio or by threshold. Default: ratio")

add_arg('ratio',            float,  0.55,               "The ratio to set zeros, the smaller portion will be zeros. Default: 0.55")
add_arg('num_epochs',       int,  120,               "The number of total epochs. Default: 120")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
parser.add_argument('--sparse_block', nargs='+', type=int, default=[1, 1], help="There must be two integers inside this array. The array defines the shape of the block, the values within which are either sparsified to all zeros or kept original. [1, 1] means unstructured pruning. Default: [1, 1]")
add_arg('data',             str, "imagenet",                 "Which data to use. 'mnist', 'cifar10' or 'imagenet'. Default: imagenet")
add_arg('log_period',       int, 100,                 "Log period in batches. Default: 100")
add_arg('test_period',      int, 5,                 "Test period in epoches. Default: 5")
add_arg('model_path',       str, "./models",         "The path to save model. Default: ./models")
add_arg('model_period',     int, 10,             "The period to save model in epochs. Default: 10")
add_arg('last_epoch',     int, -1,             "The last epoch we could train from. Default: -1")
add_arg('stable_epochs',    int, 0,              "The epoch numbers used to stablize the model before pruning. Default: 0")
add_arg('pruning_epochs',   int, 60,             "The epoch numbers used to prune the model by a ratio step. Default: 60")
add_arg('tunning_epochs',   int, 60,             "The epoch numbers used to tune the after-pruned models. Default: 60")
add_arg('pruning_steps',    int, 120,        "How many times you want to increase your ratio during training. Default: 120")
add_arg('initial_ratio',    float, 0.15,         "The initial pruning ratio used at the start of pruning stage. Default: 0.15")
add_arg('prune_params_type', str, None,           "Which kind of params should be pruned, we only support None (all but norms) and conv1x1_only for now. Default: None")
add_arg('local_sparsity', bool, False,            "Whether to prune all the parameter matrix at the same ratio or not. Default: False")
add_arg('ce_test',                 bool,   False,                                        "Whether to CE test. Default: False")
add_arg('num_workers',      int, 32,              "number of workers when loading dataset. Default: 32")
# yapf: enable

model_list = models.__all__


def piecewise_decay(args, step_per_epoch):
    bd = [step_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    last_iter = (1 + args.last_epoch) * step_per_epoch
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, last_epoch=last_iter)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def cosine_decay(args, step_per_epoch):
    last_iter = (1 + args.last_epoch) * step_per_epoch
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr,
        T_max=args.num_epochs * step_per_epoch,
        last_epoch=last_iter)
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


def create_unstructured_pruner(train_program, args, place, configs):
    if configs is None:
        return UnstructuredPruner(
            train_program,
            mode=args.pruning_mode,
            ratio=args.ratio,
            threshold=args.threshold,
            prune_params_type=args.prune_params_type,
            place=place,
            local_sparsity=args.local_sparsity,
            sparse_block=args.sparse_block)
    else:
        return GMPUnstructuredPruner(
            train_program,
            ratio=args.ratio,
            prune_params_type=args.prune_params_type,
            place=place,
            local_sparsity=args.local_sparsity,
            sparse_block=args.sparse_block,
            configs=configs)


def compress(args):
    shuffle = True
    if args.ce_test:
        # set seed
        seed = 111
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        args.num_workers = 0
        shuffle = False

    env = os.environ
    num_trainers = int(env.get('PADDLE_TRAINERS_NUM', 1))
    use_data_parallel = num_trainers > 1

    if use_data_parallel:
        # Fleet step 1: initialize the distributed environment
        role = fleet.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

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
    elif args.data == "cifar10":
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = paddle.vision.datasets.Cifar10(
            mode="train", backend="cv2", transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode="test", backend="cv2", transform=transform)
        class_dim = 10
        image_shape = "3, 32, 32"
        args.pretrained_model = False
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
    if args.use_gpu:
        places = paddle.static.cuda_places()
    else:
        places = paddle.static.cpu_places()
    place = places[0]
    exe = paddle.static.Executor(place)

    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    batch_size_per_card = args.batch_size
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=batch_size_per_card,
        shuffle=shuffle,
        drop_last=True)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=place,
        batch_sampler=batch_sampler,
        feed_list=[image, label],
        return_list=False,
        use_shared_memory=True,
        num_workers=args.num_workers)

    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        use_shared_memory=True,
        batch_size=args.batch_size_for_validation,
        shuffle=False)

    step_per_epoch = int(
        np.ceil(len(train_dataset) * 1. / args.batch_size / num_trainers))

    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    if args.data == 'cifar10':
        label = paddle.reshape(label, [-1, 1])
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    val_program = paddle.static.default_main_program().clone(for_test=True)

    opt, learning_rate = create_optimizer(args, step_per_epoch)

    # Fleet step 2: distributed strategy
    if use_data_parallel:
        dist_strategy = DistributedStrategy()
        dist_strategy.sync_batch_norm = False
        dist_strategy.exec_strategy = paddle.static.ExecutionStrategy()
        dist_strategy.fuse_all_reduce_ops = False

    train_program = paddle.static.default_main_program()

    if args.pruning_strategy == 'gmp':
        # GMP pruner step 0: define configs for GMP, no need to define configs for the base training.
        configs = {
            'stable_iterations': args.stable_epochs * step_per_epoch,
            'pruning_iterations': args.pruning_epochs * step_per_epoch,
            'tunning_iterations': args.tunning_epochs * step_per_epoch,
            'resume_iteration': (args.last_epoch + 1) * step_per_epoch,
            'pruning_steps': args.pruning_steps,
            'initial_ratio': args.initial_ratio,
        }
    elif args.pruning_strategy == 'base':
        configs = None

    # GMP pruner step 1: initialize a pruner object by calling entry function.
    pruner = create_unstructured_pruner(
        train_program, args, place, configs=configs)

    if use_data_parallel:
        # Fleet step 3: decorate the origial optimizer and minimize it
        opt = fleet.distributed_optimizer(opt, strategy=dist_strategy)
    opt.minimize(avg_cost, no_grad_set=pruner.no_grad_set)

    exe.run(paddle.static.default_startup_program())
    if args.last_epoch > -1:
        assert args.checkpoint is not None and os.path.exists(
            args.checkpoint), "Please specify a valid checkpoint path."
        paddle.static.load(train_program, args.checkpoint)

    elif args.pretrained_model:
        assert os.path.exists(
            args.
            pretrained_model), "Pretrained model path {} doesn't exist".format(
                args.pretrained_model)

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        _logger.info("Load pretrained model from {}".format(
            args.pretrained_model))
        # NOTE: We are using paddle.static.load_vars() because the pretrained model is from an older version which requires this API. 
        # Please consider using paddle.static.load(program, model_path) when possible
        paddle.static.load_vars(exe, args.pretrained_model, predicate=if_exist)

    def test(epoch, program):
        acc_top1_ns = []
        acc_top5_ns = []

        _logger.info(
            "The current sparsity of the inference model is {}%".format(
                round(100 * UnstructuredPruner.total_sparse(
                    paddle.static.default_main_program()), 2)))
        for batch_id, data in enumerate(valid_loader):
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

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))

    def train(epoch, program):
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for batch_id, data in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            loss_n, acc_top1_n, acc_top5_n = exe.run(
                program,
                feed=data,
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            # GMP pruner step 2: step() to update ratios and other internal states of the pruner.
            pruner.step()
            train_run_cost += time.time() - train_start
            total_samples += args.batch_size
            loss_n = np.mean(loss_n)
            acc_top1_n = np.mean(acc_top1_n)
            acc_top5_n = np.mean(acc_top5_n)
            if batch_id % args.log_period == 0:
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {}; acc_top1: {}; acc_top5: {}; avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch, batch_id,
                           learning_rate.get_lr(), loss_n, acc_top1_n,
                           acc_top5_n, train_reader_cost / args.log_period, (
                               train_reader_cost + train_run_cost
                           ) / args.log_period, total_samples / args.log_period,
                           total_samples / (train_reader_cost + train_run_cost
                                            )))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            learning_rate.step()
            reader_start = time.time()

    if use_data_parallel:
        # Fleet step 4: get the compiled program from fleet
        compiled_train_program = fleet.main_program
    else:
        compiled_train_program = paddle.static.CompiledProgram(
            paddle.static.default_main_program())

    for i in range(args.last_epoch + 1, args.num_epochs):
        train(i, compiled_train_program)
        # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation. 
        pruner.update_params()

        _logger.info("The current sparsity of the pruned model is: {}%".format(
            round(100 * UnstructuredPruner.total_sparse(
                paddle.static.default_main_program()), 2)))

        if (i + 1) % args.test_period == 0:
            test(i, val_program)
        if (i + 1) % args.model_period == 0:
            if use_data_parallel:
                fleet.save_persistables(executor=exe, dirname=args.model_path)
            else:
                paddle.static.load(paddle.static.default_main_program(),
                                   args.model_path)


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

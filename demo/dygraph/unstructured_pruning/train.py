import paddle
import os
import sys
import argparse
import numpy as np
from paddleslim import UnstructuredPruner, make_unstructured_pruner
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
import paddle.nn.functional as F
import functools
from paddle.vision.models import mobilenet_v1
import time
import logging
from paddleslim.common import get_logger
import paddle.distributed as dist
from paddle.distributed import ParallelEnv

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64,                 "Minibatch size. Default: 64")
add_arg('batch_size_for_validation',       int,  64,                 "Minibatch size for validation. Default: 64")
add_arg('lr',               float,  0.05,               "The learning rate used to fine-tune pruned model. Default: 0.05")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy. Default: piecewise_decay")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter. Default: 3e-5")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate. Default: 0.9")
add_arg('ratio',            float,  0.55,               "The ratio to set zeros, the smaller part bounded by the ratio will be zeros. Default: 0.55")
add_arg('pruning_mode',            str,  'ratio',               "the pruning mode: whether by ratio or by threshold. Default: ratio")
add_arg('threshold',            float,  0.01,               "The threshold to set zeros. Default: 0.01")
add_arg('num_epochs',       int,  120,               "The number of total epochs. Default: 120")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('data',             str, "imagenet",                 "Which data to use. 'cifar10' or 'imagenet'. Default: imagenet")
add_arg('log_period',       int, 100,                 "Log period in batches. Default: 100")
add_arg('test_period',      int, 5,                 "Test period in epoches. Default: 5")
add_arg('pretrained_model', str, None,              "The pretrained model the load. Default: None.")
add_arg('checkpoint',       str, None,              "The checkpoint path to resume training. Default: None.")
add_arg('model_path',       str, "./models",         "The path to save model. Default: ./models")
add_arg('model_period',     int, 10,             "The period to save model in epochs.")
add_arg('last_epoch',     int, -1,             "The last epoch we'll train from. Default: -1")
add_arg('num_workers',     int, 16,             "number of workers when loading dataset. Default: 16")
add_arg('stable_epochs',    int, 0,              "The epoch numbers used to stablize the model before pruning. Default: 0")
add_arg('pruning_epochs',   int, 60,             "The epoch numbers used to prune the model by a ratio step. Default: 60")
add_arg('tunning_epochs',   int, 60,             "The epoch numbers used to tune the after-pruned models. Default: 60")
add_arg('pruning_steps', int, 100,        "How many times you want to increase your ratio during training. Default: 100")
add_arg('initial_ratio',    float, 0.15,         "The initial pruning ratio used at the start of pruning stage. Default: 0.15")
add_arg('pruning_strategy', str, 'base',         "Which training strategy to use in pruning, we only support base and gmp for now. Default: base")
add_arg('skip_params_type', str, None,           "Which kind of params should be skipped, we only support exclude_conv1x1 for now. Default: None")
# yapf: enable


def piecewise_decay(args, step_per_epoch, model):
    bd = [step_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    last_iter = (1 + args.last_epoch) * step_per_epoch
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, last_epoch=last_iter)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=model.parameters())
    return optimizer, learning_rate


def cosine_decay(args, step_per_epoch, model):
    last_iter = (1 + args.last_epoch) * step_per_epoch
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr,
        T_max=args.num_epochs * step_per_epoch,
        last_epoch=last_iter)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=model.parameters())
    return optimizer, learning_rate


def create_optimizer(args, step_per_epoch, model):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args, step_per_epoch, model)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args, step_per_epoch, model)


def compress(args):
    place = paddle.set_device('gpu')

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1
    if use_data_parallel:
        dist.init_parallel_env()

    train_reader = None
    test_reader = None
    if args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
    elif args.data == "cifar10":
        normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='CHW')
        transform = T.Compose([T.Transpose(), normalize])
        train_dataset = paddle.vision.datasets.Cifar10(
            mode='train', backend='cv2', transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode='test', backend='cv2', transform=transform)
        class_dim = 10
    else:
        raise ValueError("{} is not supported.".format(args.data))

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=place,
        batch_sampler=batch_sampler,
        return_list=True,
        num_workers=args.num_workers,
        use_shared_memory=True)

    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        drop_last=False,
        return_list=True,
        batch_size=args.batch_size_for_validation,
        shuffle=False,
        use_shared_memory=True)
    step_per_epoch = int(
        np.ceil(len(train_dataset) / args.batch_size / ParallelEnv().nranks))
    # model definition
    model = mobilenet_v1(num_classes=class_dim, pretrained=True)
    if ParallelEnv().nranks > 1:
        model = paddle.DataParallel(model)

    opt, learning_rate = create_optimizer(args, step_per_epoch, model)

    if args.checkpoint is not None and args.last_epoch > -1:
        if args.checkpoint.endswith('pdparams'):
            args.checkpoint = args.checkpoint[:-9]
        if args.checkpoint.endswith('pdopt'):
            args.checkpoint = args.checkpoint[:-6]
        model.set_state_dict(paddle.load(args.checkpoint + ".pdparams"))
        opt.set_state_dict(paddle.load(args.checkpoint + ".pdopt"))
    elif args.pretrained_model is not None:
        if args.pretrained_model.endswith('pdparams'):
            args.pretrained_model = args.pretrained_model[:-9]
        if args.pretrained_model.endswith('pdopt'):
            args.pretrained_model = args.pretrained_model[:-6]
        model.set_state_dict(paddle.load(args.pretrained_model + ".pdparams"))

    def test(epoch):
        model.eval()
        acc_top1_ns = []
        acc_top5_ns = []
        for batch_id, data in enumerate(valid_loader):
            start_time = time.time()
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            if args.data == 'cifar10':
                y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc_top1 = paddle.metric.accuracy(logits, y_data, k=1)
            acc_top5 = paddle.metric.accuracy(logits, y_data, k=5)
            end_time = time.time()
            if batch_id % args.log_period == 0:
                _logger.info(
                    "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id,
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1.numpy()))
            acc_top5_ns.append(np.mean(acc_top5.numpy()))

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(
                acc_top1_ns, dtype="object")),
            np.mean(np.array(
                acc_top5_ns, dtype="object"))))

    def train(epoch):
        model.train()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        for batch_id, data in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            if args.data == 'cifar10':
                y_data = paddle.unsqueeze(y_data, 1)

            train_start = time.time()
            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc_top1 = paddle.metric.accuracy(logits, y_data, k=1)
            acc_top5 = paddle.metric.accuracy(logits, y_data, k=5)

            loss.backward()
            opt.step()
            learning_rate.step()
            opt.clear_grad()
            # GMP pruner step 2: step() to update ratios and other internal states of the pruner.
            pruner.step()

            train_run_cost += time.time() - train_start
            total_samples += args.batch_size

            if batch_id % args.log_period == 0:
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {}; acc_top1: {}; acc_top5: {}; avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch, batch_id,
                           opt.get_lr(),
                           np.mean(loss.numpy()),
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), train_reader_cost /
                           args.log_period, (train_reader_cost + train_run_cost
                                             ) / args.log_period, total_samples
                           / args.log_period, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0

            reader_start = time.time()

    if args.pruning_strategy == 'gmp':
        # GMP pruner step 0: define configs. No need to do this if you are not using 'gmp'
        configs = {
            'pruning_strategy': 'gmp',
            'stable_iterations': args.stable_epochs * step_per_epoch,
            'pruning_iterations': args.pruning_epochs * step_per_epoch,
            'tunning_iterations': args.tunning_epochs * step_per_epoch,
            'resume_iteration': (args.last_epoch + 1) * step_per_epoch,
            'pruning_steps': args.pruning_steps,
            'initial_ratio': args.initial_ratio,
        }
    else:
        configs = None

    # GMP pruner step 1: initialize a pruner object
    pruner = make_unstructured_pruner(
        model,
        mode=args.pruning_mode,
        ratio=args.ratio,
        threshold=args.threshold,
        skip_params_type=args.skip_params_type,
        configs=configs)

    for i in range(args.last_epoch + 1, args.num_epochs):
        train(i)
        # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation.
        pruner.update_params()

        if (i + 1) % args.test_period == 0:
            _logger.info(
                "The current sparsity of the pruned model is: {}%".format(
                    round(100 * UnstructuredPruner.total_sparse(model), 2)))
            test(i)

        if (i + 1) % args.model_period == 0:
            pruner.update_params()
            paddle.save(model.state_dict(),
                        os.path.join(args.model_path, "model.pdparams"))
            paddle.save(opt.state_dict(),
                        os.path.join(args.model_path, "model.pdopt"))


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
from paddleslim.prune.unstructured_pruner import UnstructuredPruner
from paddleslim.common import get_logger
import models
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pruned_model', str,  "models",                "Whether to use pretrained model.")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'.")
add_arg('log_period',       int, 100,                 "Log period in batches.")
# yapf: enable

model_list = models.__all__


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
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    place = places[0]
    exe = paddle.static.Executor(place)
    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    batch_size_per_card = int(args.batch_size / len(places))
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        use_shared_memory=True,
        batch_size=batch_size_per_card,
        shuffle=False)

    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    val_program = paddle.static.default_main_program().clone(for_test=True)

    exe.run(paddle.static.default_startup_program())

    if args.pruned_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pruned_model, var.name))

        _logger.info("Load pruned model from {}".format(args.pruned_model))
        paddle.static.load_vars(exe, args.pruned_model, predicate=if_exist)

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

    test(0, val_program)


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

import paddle
import os
import sys
import argparse
import numpy as np
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
from paddleslim import UnstructuredPruner
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
import paddle.nn.functional as F
import functools
from paddle.vision.models import mobilenet_v1
import time
import logging
from paddleslim.common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pruned_model', str,  "dymodels/model.pdparams",                "Whether to use pretrained model.")
add_arg('data',             str, "cifar10",                 "Which data to use. 'cifar10' or 'imagenet'.")
add_arg('log_period',       int, 100,                 "Log period in batches.")
# yapf: enable


def compress(args):
    test_reader = None
    if args.data == "imagenet":
        import imagenet_reader as reader
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
    elif args.data == "cifar10":
        normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='CHW')
        transform = T.Compose([T.Transpose(), normalize])
        val_dataset = paddle.vision.datasets.Cifar10(
            mode='test', backend='cv2', transform=transform)
        class_dim = 10
    else:
        raise ValueError("{} is not supported.".format(args.data))

    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=places,
        drop_last=False,
        return_list=True,
        batch_size=args.batch_size,
        shuffle=False,
        use_shared_memory=True)

    # model definition
    model = mobilenet_v1(num_classes=class_dim, pretrained=True)

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

    model.set_state_dict(paddle.load(args.pruned_model))
    _logger.info("The current sparsity of the pruned model is: {}%".format(
        round(100 * UnstructuredPruner.total_sparse(model), 2)))
    test(0)


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import QuantizationConfig, DistillationConfig, TrainConfig
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('model_filename',              str,    None,         "inference model filename.")
add_arg('params_filename',             str,    None,         "inference params filename.")
add_arg('save_dir',                    str,    None,         "directory to save compressed model.")
add_arg('devices',                     str,    'gpu',        "which device used to compress.")
add_arg('batch_size',                  int,    1,            "train batch size.")
add_arg('distill_loss',                str,    'l2_loss',    "which loss to used in distillation.")
add_arg('distill_node_pair',           str,    None,         "distill node pair name list.", nargs="+")
add_arg('distill_lambda',              float,  1.0,          "weight of distill loss.")
add_arg('teacher_model_dir',           str,    None,         "teacher model directory.")
add_arg('teacher_model_filename',      str,    None,         "teacher model filename.")
add_arg('teacher_params_filename',     str,    None,         "teacher params filename.")
add_arg('epochs',                      int,    3,            "train epochs.")
add_arg('optimizer',                   str,    'SGD',        "optimizer to used.")
add_arg('learning_rate',               float,  0.0001,       "learning rate in optimizer.")
add_arg('weight_decay',                float,  0.0001,       "weight decay in optimizer.")
add_arg('eval_iter',                   int,    1000,         "how many iteration to eval.")
add_arg('origin_metric',               float,  None,         "metric of inference model to compressed.")
# yapf: enable


def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.float32([item[0] for item in data])
            yield {"inputs": imgs}

    return gen


def eval_function(exe, place, compiled_test_program, test_feed_names,
                  test_fetch_list):
    val_reader = paddle.batch(reader.val(), batch_size=1)
    image = paddle.static.data(
        name='x', shape=[None, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    results = []
    for batch_id, data in enumerate(val_reader()):
        # top1_acc, top5_acc
        if len(test_feed_names) == 1:
            # eval "infer model", which input is image, output is classification probability
            image = data[0][0].reshape((1, 3, 224, 224))
            label = [[d[1]] for d in data]
            pred = exe.run(compiled_test_program,
                           feed={test_feed_names[0]: image},
                           fetch_list=test_fetch_list)
            pred = np.array(pred[0])
            label = np.array(label)
            sort_array = pred.argsort(axis=1)
            top_1_pred = sort_array[:, -1:][:, ::-1]
            top_1 = np.mean(label == top_1_pred)
            top_5_pred = sort_array[:, -5:][:, ::-1]
            acc_num = 0
            for i in range(len(label)):
                if label[i][0] in top_5_pred[i]:
                    acc_num += 1
            top_5 = float(acc_num) / len(label)
            results.append([top_1, top_5])
        else:
            # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
            image = data[0][0].reshape((1, 3, 224, 224))
            label = [[d[1]] for d in data]
            result = exe.run(
                compiled_test_program,
                feed={test_feed_names[0]: image,
                      test_feed_names[1]: label},
                fetch_list=test_fetch_list)
            result = [np.mean(r) for r in result]
            results.append(result)
    result = np.mean(np.array(results), axis=0)
    return result[0]


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()

    default_qat_config = {
        "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],
        "weight_bits": 8,
        "activation_bits": 8,
        "is_full_quantize": False,
        "not_quant_pattern": ["skip_quant"],
    }

    default_distill_config = {
        "distill_loss": args.distill_loss,
        "distill_node_pair": args.distill_node_pair,
        "distill_lambda": args.distill_lambda,
        "teacher_model_dir": args.teacher_model_dir,
        "teacher_model_filename": args.teacher_model_filename,
        "teacher_params_filename": args.teacher_params_filename,
    }

    default_train_config = {
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "optim_args": {
            "weight_decay": args.weight_decay
        },
        "eval_iter": args.eval_iter,
        "origin_metric": args.origin_metric
    }

    train_reader = paddle.batch(reader.train(), batch_size=64)
    train_dataloader = reader_wrapper(train_reader)

    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        strategy_config={
            "QuantizationConfig": QuantizationConfig(**default_qat_config),
            "DistillationConfig": DistillationConfig(**default_distill_config)
        },
        train_config=TrainConfig(**default_train_config),
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        devices=args.devices)

    ac.compression()

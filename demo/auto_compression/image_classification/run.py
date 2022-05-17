import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.auto_compression.config_helpers import load_config
from paddleslim.auto_compression import AutoCompression
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('model_filename',              str,    None,         "inference model filename.")
add_arg('params_filename',             str,    None,         "inference params filename.")
add_arg('save_dir',                    str,    None,         "directory to save compressed model.")
add_arg('batch_size',                  int,    1,            "train batch size.")
add_arg('config_path',                 str,    None,         "path of compression strategy config.")
add_arg('data_dir',                    str,    None,         "path of dataset")


# yapf: enable
def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.float32([item[0] for item in data])
            yield {"x": imgs}

    return gen


def eval_reader(data_dir, batch_size):
    val_reader = paddle.batch(
        reader.val(data_dir=data_dir), batch_size=batch_size)
    return val_reader


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    val_reader = eval_reader(data_dir, batch_size=1)
    image = paddle.static.data(
        name='x', shape=[None, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    results = []
    for batch_id, data in enumerate(val_reader()):
        # top1_acc, top5_acc
        if len(test_feed_names) == 1:
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
        if batch_id % 5000 == 0:
            print('Eval iter: ', batch_id)
    result = np.mean(np.array(results), axis=0)
    return result[0]


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()
    compress_config, train_config = load_config(args.config_path)
    data_dir = args.data_dir

    train_reader = paddle.batch(
        reader.train(data_dir=data_dir), batch_size=args.batch_size)
    train_dataloader = reader_wrapper(train_reader)

    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        strategy_config=compress_config,
        train_config=train_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        eval_dataloader=reader_wrapper(eval_reader(data_dir, 64)))

    ac.compress()

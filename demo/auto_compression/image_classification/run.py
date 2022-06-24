import os
import sys
sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config
from paddleslim.auto_compression import AutoCompression
from utility import add_arguments, print_arguments

def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    return parser


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')
    
# yapf: enable
def reader_wrapper(reader, input_name):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.float32([item[0] for item in data])
            yield {input_name: imgs}

    return gen


def eval_reader(data_dir, batch_size):
    val_reader = paddle.batch(
        reader.val(data_dir=data_dir), batch_size=batch_size)
    return val_reader


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    val_reader = eval_reader(data_dir, batch_size=global_config['batch_size'])
    image = paddle.static.data(
        name=global_config['input_name'], shape=[None, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    results = []
    for batch_id, data in enumerate(val_reader()):
        # top1_acc, top5_acc
        if len(test_feed_names) == 1:
            image = np.array([[d[0]] for d in data])
            image = image.reshape((len(data), 3, 224, 224))
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
            image = np.array([[d[0]] for d in data])
            image = image.reshape((len(data), 3, 224, 224))
            label = [[d[1]] for d in data]
            result = exe.run(
                compiled_test_program,
                feed={test_feed_names[0]: image,
                      test_feed_names[1]: label},
                fetch_list=test_fetch_list)
            result = [np.mean(r) for r in result]
            results.append(result)
        if batch_id % 50 == 0:
            print('Eval iter: ', batch_id)
    result = np.mean(np.array(results), axis=0)
    return result[0]


def main():
    global global_config
    compress_config, train_config, global_config = load_slim_config(
        args.config_path)
    global data_dir
    data_dir = global_config['data_dir']

    train_reader = paddle.batch(
        reader.train(data_dir=data_dir), batch_size=global_config['batch_size'])
    train_dataloader = reader_wrapper(train_reader, global_config['input_name'])

    ac = AutoCompression(
        model_dir=global_config['model_dir'],
        model_filename=global_config['model_filename'],
        params_filename=global_config['params_filename'],
        save_dir=args.save_dir,
        strategy_config=compress_config,
        train_config=train_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        eval_dataloader=reader_wrapper(eval_reader(data_dir, global_config['batch_size']), global_config['input_name']))

    ac.compress()
    
if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    print_arguments(args)
    main()

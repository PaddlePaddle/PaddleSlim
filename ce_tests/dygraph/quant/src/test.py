from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import sys
import argparse
import functools
import math

import paddle
import paddle.inference as paddle_infer
from utility import add_arguments, print_arguments
import imagenet_dataset as dataset


def eval(args):
    # create predictor
    model_file = os.path.join(args.model_path, args.model_filename)
    params_file = os.path.join(args.model_path, args.params_filename)
    config = paddle_infer.Config(model_file, params_file)
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    if not args.ir_optim:
        config.switch_ir_optim(False)

    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])

    # prepare data
    val_dataset = dataset.ImageNetDataset(
        path=os.path.join(args.data_dir, 'val'), mode='val')
    eval_loader = paddle.io.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=5)

    cost_time = 0.
    total_num = 0.
    correct_1_num = 0
    correct_5_num = 0
    for batch_id, data in enumerate(eval_loader()):
        # set input
        img_np = np.array([tensor.numpy() for tensor in data[0]])
        label_np = np.array([tensor.numpy() for tensor in data[1]])

        input_handle.reshape(img_np.shape)
        input_handle.copy_from_cpu(img_np)

        # run
        t1 = time.time()
        predictor.run()
        t2 = time.time()
        cost_time += (t2 - t1)

        output_data = output_handle.copy_to_cpu()

        # calculate accuracy
        for i in range(len(label_np)):
            label = label_np[i][0]
            result = output_data[i, :]
            index = result.argsort()
            total_num += 1
            if index[-1] == label:
                correct_1_num += 1
            if label in index[-5:]:
                correct_5_num += 1

        if batch_id % 10 == 0:
            acc1 = correct_1_num / total_num
            acc5 = correct_5_num / total_num
            avg_time = cost_time / total_num
            print(
                "batch_id {}, acc1 {:.3f}, acc5 {:.3f}, avg time {:.5f} sec/img".
                format(batch_id, acc1, acc5, avg_time))

        if args.test_samples > 0 and \
            (batch_id + 1)* args.batch_size >= args.test_samples:
            break

    acc1 = correct_1_num / total_num
    acc5 = correct_5_num / total_num
    avg_time = cost_time / total_num
    print("End test: test image {}".format(total_num))
    print("test_acc1 {:.4f}, test_acc5 {:.4f}, avg time {:.5f} sec/img".format(
        acc1, acc5, avg_time))
    print("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('model_path', str, "", "The inference model path.")
    add_arg('model_filename', str, "model.pdmodel", "model filename")
    add_arg('params_filename', str, "model.pdiparams", "params filename")
    add_arg('data_dir', str, "/dataset/ILSVRC2012/",
            "The ImageNet dataset root dir.")
    add_arg('test_samples', int, -1,
            "Test samples. If set -1, use all test samples")
    add_arg('batch_size', int, 10, "Batch size.")
    add_arg('use_gpu', bool, False, "Use gpu.")
    add_arg('ir_optim', bool, False, "Enable ir optim.")

    args = parser.parse_args()
    print_arguments(args)

    eval(args)


if __name__ == '__main__':
    main()

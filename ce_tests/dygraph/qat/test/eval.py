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
import paddle.fluid as fluid
import reader as reader
from utility import add_arguments, print_arguments

def eval(args):
    data_dir = args.data_dir

    paddle.enable_static()
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
	
    [inference_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(args.model_path, exe, 
                    args.model_filename, args.params_filename)
    
    feed_vars = [fluid.framework._get_var(str(var_name), inference_program) \
            for var_name in feed_target_names]
    val_reader = paddle.batch(reader.val(data_dir=data_dir), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=feed_vars)

    total = 0
    correct = 0
    correct_5 = 0
    for batch_id, data in enumerate(val_reader()):
        labels = []
        in_data = []
        for dd in data:
            labels.append(dd[1])
            img_np = np.array(dd[0])
            in_data.append([img_np])
            
        t1 = time.time()
        fetch_out = exe.run(inference_program, fetch_list=fetch_targets,
                            feed=feeder.feed(in_data))
        t2 = time.time()
        
        for i in range(len(labels)):
            label = labels[i]
            result = np.array(fetch_out[0][i])
            index = result.argsort()
            top_1_index = index[-1]
            top_5_index = index[-5:]
            total += 1
            if top_1_index == label:
                correct += 1
            if label in top_5_index:
                correct_5 += 1

        if batch_id % 10 == 0:
            acc1 = float(correct) / float(total)
            acc5 = float(correct_5) / float(total)
            period = t2 - t1
            print("Testbatch {0}, "
                  "acc1 {1}, acc5 {2}, time {3}".format(batch_id, \
                  acc1, acc5, "%2.2f sec" % period))
        
        if args.test_samples > 0 and \
            (batch_id + 1)* args.batch_size >= args.test_samples:
            break
    
    acc1 = correct / total
    acc5 = correct_5 / total
    print("End test: test_acc1 {0}, test_acc5 {1}".format(acc1, acc5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('model_path',      str,   "",                 "The inference model path.")
    add_arg('model_filename',  str,   "int8_infer.pdmodel", "model filename")
    add_arg('params_filename',  str,  "int8_infer.pdiparams","params filename")
    add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
    add_arg('data_dir',         str,  "/dataset/ILSVRC2012/",  "The ImageNet dataset root dir.")
    add_arg('test_samples',     int,  -1,                "Test samples. If set -1, use all test samples")
    add_arg('batch_size',       int,  20,                "Batch size.")

    args = parser.parse_args()
    print_arguments(args)
    
    eval(args)


if __name__ == '__main__': 
    main()

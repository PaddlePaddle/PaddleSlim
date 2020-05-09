# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import math
import time
import random
import argparse
import multiprocessing

import numpy as np
import paddle
import paddle.fluid as fluid

import reader
import model_utils
import creator
from eval import test_process
from models.model_check import check_cuda
from models.model_check import check_version

# the function to train model
def do_train(args):
    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        dev_count = min(multiprocessing.cpu_count(), args.cpu_num)
        if (dev_count < args.cpu_num):
            print(
                "WARNING: The total CPU NUM in this machine is %d, which is less than cpu_num parameter you set. "
                "Change the cpu_num from %d to %d" %
                (dev_count, args.cpu_num, dev_count))
        os.environ['CPU_NUM'] = str(dev_count)
        place = fluid.CPUPlace()

    train_program = fluid.Program()
    test_program = fluid.Program()
    startup_program = fluid.Program()

    dataset = reader.Dataset(args)
    with fluid.program_guard(train_program, startup_program):
        #train_program.random_seed = args.random_seed
        startup_program.random_seed = args.random_seed

        with fluid.unique_name.guard():
            train_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='train')

            optimizer = fluid.optimizer.Adam(
                learning_rate=args.base_learning_rate)
            optimizer.minimize(train_ret["avg_cost"])
            
    with fluid.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            test_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='test')

            test_program = test_program.clone(for_test=True)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.init_checkpoint:
        model_utils.init_checkpoint(exe, args.init_checkpoint, train_program)
    if dev_count > 1:
        device = "GPU" if args.use_cuda else "CPU"
        print("%d %s are used to train model" % (dev_count, device))
        # multi cpu/gpu config
        exec_strategy = fluid.ExecutionStrategy()

        build_strategy = fluid.compiler.BuildStrategy()

        compiled_prog = fluid.compiler.CompiledProgram(
            train_program).with_data_parallel(
                loss_name=train_ret['avg_cost'].name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
    else:
        compiled_prog = fluid.compiler.CompiledProgram(train_program)

    # start training
    num_train_examples = dataset.get_num_examples(args.train_data)
    max_train_steps = args.epoch * num_train_examples // args.batch_size
    print("Num train examples: %d" % num_train_examples)
    print("Max train steps: %d" % max_train_steps)

    train_generator = creator.create_lexnet_data_generator(args,
                                         reader=dataset, 
                                         file_name=args.train_data,
                                         place=place, 
                                         mode='train')
    test_generator = creator.create_lexnet_data_generator(args, 
                                         reader=dataset,
                                         file_name=args.test_data, 
                                         place=place, 
                                         mode='test')

    train_reader, test_reader = train_ret['pyreader'], test_ret['pyreader']
    train_reader.set_batch_generator(train_generator, places=place)
    test_reader.set_batch_generator(test_generator, places=place)

    ce_info = []
    step = 0
    ce_time = 0
    train_reader.start()
    while True:
        try:
            # this is for minimizing the fetching op, saving the training speed.
            if step % args.print_steps == 0:
                fetch_list = [
                    train_ret["avg_cost"], train_ret["precision"],
                    train_ret["recall"], train_ret["f1_score"],
                            train_ret["crf_avg_cost"], train_ret["teacher_cost"]
                ]
            else:
                fetch_list = []

            start_time = time.time()
            outputs = exe.run(
            program=compiled_prog,
            fetch_list=fetch_list)

            end_time = time.time()
            if step % args.print_steps == 0:
                avg_cost, precision, recall, f1_score, crf_avg_cost, teacher_cost = [
                    np.mean(x) for x in outputs
                ]
                print("Data loader queue size: %d " % train_reader.queue.size())
                print(
                    "[train] step = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, crf_avg_cost: %.5f, teacher_cost: %.5f, elapsed time %.5f"
                    % (step, avg_cost, precision, recall, f1_score, crf_avg_cost, teacher_cost,
                    end_time - start_time))

            if step % args.validation_steps == 0:
                test_process(exe, test_program, test_reader, test_ret)

                ce_time += end_time - start_time
                ce_info.append([ce_time, avg_cost, precision, recall, f1_score])

            # save checkpoints
            if step % args.save_steps == 0 and step != 0:
                save_path = os.path.join(args.model_save_dir,
                            "step_" + str(step))
                fluid.io.save_persistables(exe, save_path, train_program)
            step += 1
        except fluid.core.EOFException:
            train_reader.reset()
            break

    if args.enable_ce:
        card_num = get_cards()
        ce_cost = 0
        ce_f1 = 0
        ce_p = 0
        ce_r = 0
        ce_time = 0
        try:
            ce_time = ce_info[-2][0]
            ce_cost = ce_info[-2][1]
            ce_p = ce_info[-2][2]
            ce_r = ce_info[-2][3]
            ce_f1 = ce_info[-2][4]
        except:
            print("ce info error")
        print("kpis\teach_step_duration_card%s\t%s" % (card_num, ce_time))
        print("kpis\ttrain_cost_card%s\t%f" % (card_num, ce_cost))
        print("kpis\ttrain_precision_card%s\t%f" % (card_num, ce_p))
        print("kpis\ttrain_recall_card%s\t%f" % (card_num, ce_r))
        print("kpis\ttrain_f1_card%s\t%f" % (card_num, ce_f1))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == "__main__":

    parser = argparse.ArgumentParser(__doc__)
    model_utils.load_yaml(parser, 'conf/args.yaml')

    # config for pantheon student
    parser.add_argument('--in_path', type=str, default=None, help="The path of dumped knowledge from teacher for offline mode.")
    parser.add_argument('--in_address', type=str, default=None, help="The IP port number to receive knowledge from teacher for \
                            online mode")
    
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    check_version()
    do_train(args)

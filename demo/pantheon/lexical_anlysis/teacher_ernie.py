#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
Baidu's open-source Lexical Analysis tool for Chinese, including:
    1. Word Segmentation,
    2. Part-of-Speech Tagging
    3. Named Entity Recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import sys
from collections import namedtuple
from paddleslim.pantheon import Teacher
import paddle.fluid as fluid

import creator
import model_utils
print('model representation') 
from models.representation.ernie import ErnieConfig
print('model check') 
from models.model_check import check_cuda
from models.model_check import check_version



def do_eval(args):
    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    print('ernie config') 
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    test_program = fluid.Program()
    print('test program') 
    with fluid.program_guard(test_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():
            test_ret = creator.create_ernie_model(args, ernie_config)
    test_program = test_program.clone(for_test=True)
    #print('create pyreader') 
    pyreader = creator.create_pyreader(
        args,
        file_name=args.test_data,
        feed_list=[ret.name for ret in test_ret['feed_list']],
        model="ernie",
        place=place,
        return_reader=True,
        mode='test')

    #data_inter = reader.data_generator(args.test_data, args.batch_size, 1, shuffle=False, phase="train")

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    if not args.init_checkpoint:
        raise ValueError(
            "args 'init_checkpoint' should be set if only doing test or infer!")
    model_utils.init_checkpoint(exe, args.init_checkpoint, test_program)
    
    teacher = Teacher(out_path=None, out_port=int(args.out_port))
    teacher.start()
    print('run teacher......')
    
    test_ret["chunk_evaluator"].reset()
   
    reader_config = {"batch_generator": pyreader}

    teacher.start_knowledge_service(
            feed_list=[test_ret["words"].name, test_ret["sent_ids"].name, test_ret["pos_ids"].name, test_ret["input_mask"].name, test_ret["labels"].name, test_ret["seq_lens"].name],
            schema={"crf_decode":test_ret["crf_decode"],"seq_lens":test_ret["seq_lens"]},
            program=test_program,
            reader_config=reader_config,
            exe=exe,
            times=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    model_utils.load_yaml(parser, './conf/ernie_args.yaml')
    
    # config for pantheon teacher
    parser.add_argument('--out_path', type=str, default=None, help="The path to dump knowledge for offline mode.")
    parser.add_argument('--out_port', type=str, default=None, help="The IP port number to send out knowledge for \
                            online mode, should be unique when launching multiple teachers in \
                            the same node.")
    
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    check_version()
    model_utils.print_arguments(args)
    do_eval(args)

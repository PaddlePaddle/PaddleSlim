# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import math
import numpy as np

import paddle
import paddle.hapi as hapi
from paddle.hapi.model import Input
from paddle.metric.metrics import Accuracy
import paddle.vision.models as models

from paddleslim.dygraph.quant import QAT

import imagenet_dataset as dataset

def main():
    model_list = [x for x in models.__dict__["__all__"]]
    assert FLAGS.arch in model_list, "Expected FLAGS.arch in {}, but received {}".format(
        model_list, FLAGS.arch)
    model = models.__dict__[FLAGS.arch](pretrained=True)

    if FLAGS.enable_quant:
        print("quantize model")
        quant_config = {
            'weight_preprocess_type': None,
            'activation_preprocess_type': 'PACT' if FLAGS.use_pact else None,
            'weight_quantize_type': "channel_wise_abs_max",
            'activation_quantize_type': 'moving_average_abs_max',
            'weight_bits': 8,
            'activation_bits': 8,
            'window_size': 10000,
            'moving_rate': 0.9,
            'quantizable_layer_type': ['Conv2D', 'Linear'],}
        dygraph_qat = QAT(quant_config)
        dygraph_qat.quantize(model)

    model = hapi.Model(model)
    
    train_dataset = dataset.ImageNetDataset(data_dir=FLAGS.data, mode='train')
    val_dataset = dataset.ImageNetDataset(data_dir=FLAGS.data, mode='val')

    optim = paddle.optimizer.SGD(learning_rate=FLAGS.lr, parameters=model.parameters(),
            weight_decay=FLAGS.weight_decay)
    
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), Accuracy(topk=(1, 5)))

    checkpoint_dir = os.path.join(FLAGS.output_dir, "checkpoint", FLAGS.arch + "_checkpoint",
                              time.strftime('%Y-%m-%d-%H-%M', time.localtime()))
    model.fit(train_dataset,
              val_dataset,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epoch,
              save_dir=checkpoint_dir,
              num_workers=FLAGS.num_workers)
        
    if FLAGS.enable_quant:
        quant_output_dir = os.path.join(FLAGS.output_dir, "quant_dygraph",
                FLAGS.arch, "int8_infer")
        input_spec = paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')
        dygraph_qat.save_quantized_model(model.network, quant_output_dir, [input_spec])
        print("Save quantized inference model in " + quant_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training on ImageNet")
    
    # model
    parser.add_argument("--arch", type=str, default='mobilenet_v1', help="model arch")
    parser.add_argument("--output_dir", type=str, default='output', help="output dir")
    
    # data
    parser.add_argument('--data', default="", 
            help='path to dataset (should have subdirectories named "train" and "val"')
    
    # train
    parser.add_argument("--epoch", default=1, type=int, help="number of epoch")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="dataloader workers")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")
    
    # quant
    parser.add_argument("--enable_quant", action='store_true', help="enable quant model")
    parser.add_argument("--use_pact", action='store_true', help="use pact")

    FLAGS = parser.parse_args()
    
    main()

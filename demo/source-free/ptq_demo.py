import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

import numpy as np
from collections import namedtuple, Iterable
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import QuantizationConfig, DistillationConfig, MultiTeacherDistillationConfig, HyperParameterOptimizationConfig, TrainConfig

default_ptq_config = {
   "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],
   "weight_bits": 8,
   "activation_bits": 8,
}

default_hpo_config = {
"ptq_algo": ["KL", "hist"],
"bias_correct": [True],
"weight_quantize_type": ["channel_wise_abs_max"], 
"hist_percent": [0.999, 0.99999],
"batch_size": [4, 8, 16],
"batch_num": [4, 8, 16],
"max_quant_count": 20
}

train_reader = paddle.batch(reader.train(), batch_size=64)
eval_reader = paddle.batch(reader.val(), batch_size=64)
def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.float32([item[0] for item in data])
            yield {"inputs":imgs}
    return gen

train_dataloader = reader_wrapper(train_reader)
eval_dataloader = reader_wrapper(eval_reader)

ac = AutoCompression(model_dir='MobileNetV3_small_x1_0_ssld_infer', 
                     model_filename='inference.pdmodel', 
                     params_filename='inference.pdiparams', 
                     save_dir='./mbv3_small_ptq_hpo_output', 
                     strategy_config={"QuantizationConfig": QuantizationConfig(**default_ptq_config), 
                     "HyperParameterOptimizationConfig": HyperParameterOptimizationConfig(**default_hpo_config)}, 
                     train_config=None,
                     train_dataloader=train_dataloader, eval_callback=eval_dataloader)

ac.compression()

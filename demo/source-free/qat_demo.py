import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import QuantizationConfig, DistillationConfig, MultiTeacherDistillationConfig, HyperParameterOptimizationConfig, TrainConfig

default_qat_config = {
   "quantize_op_types": ["conv2d", "depthwise_conv2d"],
   "weight_bits": 8,
   "activation_bits": 8,
   "is_full_quantize": False,
   "not_quant_pattern": ["skip_quant"],
}

default_distill_config = {
"distill_loss": 'L2',
"distill_node_pair": ["teacher_conv2d_54.tmp_0", "conv2d_54.tmp_0", "teacher_conv2d_55.tmp_0", "conv2d_55.tmp_0",\
        "teacher_conv2d_57.tmp_0", "conv2d_57.tmp_0", "teacher_elementwise_add_0", "elementwise_add_0", \
        "teacher_conv2d_61.tmp_0", "conv2d_61.tmp_0", "teacher_elementwise_add_1", "elementwise_add_1", \
        "teacher_elementwise_add_2", "elementwise_add_2", "teacher_conv2d_67.tmp_0", "conv2d_67.tmp_0", \
        "teacher_elementwise_add_3", "elementwise_add_3", "teacher_elementwise_add_4", "elementwise_add_4", \
        "teacher_elementwise_add_5", "elementwise_add_5", "teacher_conv2d_75.tmp_0", "conv2d_75.tmp_0", \
        "teacher_elementwise_add_6", "elementwise_add_6", "teacher_elementwise_add_7", "elementwise_add_7", \
        "teacher_conv2d_81.tmp_0", "conv2d_81.tmp_0", "teacher_elementwise_add_8", "elementwise_add_8", \
        "teacher_elementwise_add_9", "elementwise_add_9", "teacher_conv2d_87.tmp_0", "conv2d_87.tmp_0", \
        "teacher_linear_1.tmp_0", "linear_1.tmp_0"],
"distill_lambda": 1.0,
"teacher_model_dir": "./MobileNetV2_ssld_infer",
"teacher_model_filename": 'inference',
"teacher_params_filename": 'inference',
}

default_train_config = {
"epochs": 1,
"optimizer": "SGD",
"learning_rate": 0.0001,
"weight_decay": 0.00004,
"eval_iter": 1000,
"origin_metric": 0.765,
}


train_reader = paddle.batch(reader.train(), batch_size=64)
def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.float32([item[0] for item in data])
            yield {"inputs":imgs}
    return gen

train_dataloader = reader_wrapper(train_reader)

def eval_function(exe, place, compiled_test_program, test_feed_names, test_fetch_list):
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
            result = exe.run(compiled_test_program,
                             feed={
                                 test_feed_names[0]: image,
                                 test_feed_names[1]: label
                             },
                             fetch_list=test_fetch_list)
            result = [np.mean(r) for r in result]
            results.append(result)
    result = np.mean(np.array(results), axis=0)
    return result[0]

paddle.set_device("gpu")
ac = AutoCompression(model_dir='./MobileNetV2_ssld_infer', 
                     model_filename='inference', 
                     params_filename='inference', 
                     save_dir='./mbv2_qat_distill_output', 
                     strategy_config={"QuantizationConfig": QuantizationConfig(**default_qat_config), 
                     "DistillationConfig": DistillationConfig(**default_distill_config)}, 
                     train_config=TrainConfig(**default_train_config),
                     train_dataloader=train_dataloader, eval_callback=eval_function)

ac.compression()

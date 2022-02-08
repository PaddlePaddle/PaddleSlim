import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

import numpy as np
from functools import partial
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall
import imagenet_reader as reader
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import QuantizationConfig, DistillationConfig, MultiTeacherDistillationConfig, HyperParameterOptimizationConfig, TrainConfig
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.metrics import Mcc, PearsonAndSpearman

default_qat_config = {
   "quantize_op_types": ["conv2d", "depthwise_conv2d"],
   "weight_bits": 8,
   "activation_bits": 8,
   "is_full_quantize": False,
   "not_quant_pattern": ["skip_quant"],
}

default_distill_config = {
"distill_loss": 'L2',
"distill_node_pair": ["teacher_tmp_9", "tmp_9", "teacher_tmp_12", "tmp_12",\
        "teacher_tmp_15", "tmp_15", "teacher_tmp_18", "tmp_18", \
        "teacher_tmp_21", "tmp_21", "teacher_tmp_24", "tmp_24", \
        "teacher_tmp_27", "tmp_27", "teacher_tmp_30", "tmp_30", \
        "teacher_tmp_33", "tmp_33", "teacher_tmp_36", "tmp_36", \
        "teacher_tmp_39", "tmp_39", "teacher_tmp_42", "tmp_42", \
        "teacher_linear_147.tmp_1", "linear_147.tmp_1"],
"distill_lambda": 1.0,
"teacher_model_dir": "./static_bert_models",
"teacher_model_filename": 'bert',
"teacher_params_filename": 'bert',
}

default_train_config = {
"epochs": 1,
"optimizer": "SGD",
"learning_rate": 0.0001,
"weight_decay": 0.00004,
"eval_iter": 1000,
"origin_metric": 0.93,
}

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "sts-b": PearsonAndSpearman,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

paddle.enable_static()
paddle.set_device("gpu")

def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """
    Convert a glue example into necessary features.
    """
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    #if (int(is_test) + len(example)) == 2:
    #    example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    #else:
    #    example = tokenizer(
    #        example['sentence1'],
    #        text_pair=example['sentence2'],
    #        max_seq_len=max_seq_length)
    example = tokenizer(example['sentence'], max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def create_data_holder(task_name):
    """
    Define the input data holder for the glue task.
    """
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    token_type_ids = paddle.static.data(
        name="token_type_ids", shape=[-1, -1], dtype="int64")
    if task_name == "sts-b":
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="float32")
    else:
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    return [input_ids, token_type_ids, label]

def reader():
    # Create the tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained('./static_bert_models/')
    train_ds = load_dataset('glue', 'sst-2', splits="train")

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=128,
        is_test=True)

    train_ds = train_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=32, shuffle=True)

    [input_ids, token_type_ids, labels] = create_data_holder('sst-2')
    feed_list_name = []
    train_data_loader = DataLoader(
        dataset=train_ds,
        feed_list=[input_ids, token_type_ids],
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    dev_trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=128)
    dev_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    dev_ds = load_dataset('glue', 'sst-2', splits='dev')
    dev_ds = dev_ds.map(dev_trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=32, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        num_workers=0,
        feed_list=[input_ids, token_type_ids, labels],
        return_list=False)

    return train_data_loader, dev_data_loader


train_dataloader, eval_dataloader = reader()
metric_class = METRIC_CLASSES['sst-2']
metric = metric_class()

def eval_function(exe, place, compiled_test_program, test_feed_names, test_fetch_list):
    metric.reset()
    for data in eval_dataloader():
        logits = exe.run(compiled_test_program,
                       feed={test_feed_names[0]: data[0]['input_ids'], test_feed_names[1]: data[0]['token_type_ids']},
                       fetch_list=test_fetch_list)
        paddle.disable_static()
        labels_pd = paddle.to_tensor(np.array(data[0]['label']))
        logits_pd = paddle.to_tensor(logits[0])
        correct = metric.compute(logits_pd, labels_pd)
        metric.update(correct)
        paddle.enable_static()
    res = metric.accumulate()
    return res


ac = AutoCompression(model_dir='./static_bert_models', 
                     model_filename='bert', 
                     params_filename='bert', 
                     save_dir='./bert_qat_distill_output', 
                     strategy_config={"QuantizationConfig": QuantizationConfig(**default_qat_config), 
                     "DistillationConfig": DistillationConfig(**default_distill_config)}, 
                     train_config=TrainConfig(**default_train_config),
                     train_dataloader=train_dataloader, eval_callback=eval_function)

ac.compression()

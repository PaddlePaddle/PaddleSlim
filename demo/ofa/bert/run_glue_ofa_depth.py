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

import argparse
import logging
import os
import random
import time
import math
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertModel, BertForSequenceClassification, BertTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
import paddlenlp.datasets as datasets
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig, utils
from paddleslim.nas.ofa.convert_super import Convert, supernet

TASK_CLASSES = {
    "cola": (datasets.GlueCoLA, Mcc),
    "sst-2": (datasets.GlueSST2, Accuracy),
    "mrpc": (datasets.GlueMRPC, AccuracyAndF1),
    "sts-b": (datasets.GlueSTSB, PearsonAndSpearman),
    "qqp": (datasets.GlueQQP, AccuracyAndF1),
    "mnli": (datasets.GlueMNLI, Accuracy),
    "qnli": (datasets.GlueQNLI, Accuracy),
    "rte": (datasets.GlueRTE, Accuracy),
}

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(TASK_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lambda_logit",
        default=1.0,
        type=float,
        help="lambda for logit loss.")
    parser.add_argument(
        "--lambda_rep",
        default=0.1,
        type=float,
        help="lambda for hidden state distillation loss.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        '--width_mult_list',
        nargs='+',
        type=float,
        default=[1.0, 5 / 6, 2 / 3, 0.5],
        help="width mult in compress")
    parser.add_argument(
        '--depth_mult_list',
        nargs='+',
        type=float,
        default=[1.0, 0.75, 0.5],
        help="width mult in compress")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def evaluate(model,
             criterion,
             metric,
             teacher_metric,
             data_loader,
             width_mult=1.0,
             depth_mult=1.0):
    with paddle.no_grad():
        model.eval()
        metric.reset()
        #teacher_metric.reset()
        for batch in data_loader:
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids, attention_mask=[None, None])
            if isinstance(logits, tuple):
                #teacher_logits = logits[1]
                logits = logits[0]
            loss = criterion(logits, labels)
            correct = metric.compute(logits, labels)
            #teacher_correct = metric.compute(teacher_logits, labels)
            metric.update(correct)
            #teacher_metric.update(teacher_correct)
        results = metric.accumulate()
        #teacher_results = teacher_metric.accumulate()
        print(
            "depth_mult: %f, width_mult: %f, eval loss: %f, %s: %s\n" %
            (depth_mult, width_mult, loss.numpy(), metric.name(), results),
            end='')
        model.train()


### monkey patch for bert forward to accept [attention_mask, head_mask] as  attention_mask
def bert_forward(self,
                 input_ids,
                 token_type_ids=None,
                 position_ids=None,
                 attention_mask=[None, None],
                 depth_mult=1.0):
    wtype = self.pooler.dense.fn.weight.dtype if hasattr(
        self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
    if attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    encoder_outputs = self.encoder(
        embedding_output, attention_mask, depth_mult=depth_mult)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output


BertModel.forward = bert_forward


def transformer_encoder_forward(self, src, src_mask=None, depth_mult=1.):
    output = src

    depth = round(self.num_layers * depth_mult)
    kept_layers_index = []
    for i in range(1, depth + 1):
        kept_layers_index.append(math.floor(i / depth_mult) - 1)

    for i in kept_layers_index:
        output = self.layers[i](output, src_mask=src_mask)

    if self.norm is not None:
        output = self.norm(output)

    return output


paddle.nn.TransformerEncoder.forward = transformer_encoder_forward


def sequence_forward(self,
                     input_ids,
                     token_type_ids=None,
                     position_ids=None,
                     attention_mask=[None, None],
                     depth=1.0):
    _, pooled_output = self.bert(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        depth_mult=depth)

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    return logits


BertForSequenceClassification.forward = sequence_forward


def soft_cross_entropy(inp, target):
    inp_likelihood = F.log_softmax(inp, axis=-1)
    target_prob = F.softmax(target, axis=-1)
    return -1. * paddle.mean(paddle.sum(inp_likelihood * target_prob, axis=-1))


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(([i] * (len(seq) + len(sep)) for i, (sep, seq) in
                           enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask for sep, seq, s_mask, mask in
                      zip(separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = dataset_class.get_datasets(['train'])

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.apply(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(),  # length
        Stack(dtype="int64" if train_ds.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples)) if i != 2]
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)
    if args.task_name == "mnli":
        dev_dataset_matched, dev_dataset_mismatched = dataset_class.get_datasets(
            ["dev_matched", "dev_mismatched"])
        dev_dataset_matched = dev_dataset_matched.apply(trans_func, lazy=True)
        dev_dataset_mismatched = dev_dataset_mismatched.apply(
            trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_dataset_matched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_dataset_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_dataset_mismatched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_dataset_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
    else:
        dev_dataset = dataset_class.get_datasets(["dev"])
        dev_dataset = dev_dataset.apply(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_dataset, batch_size=args.batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    num_labels = 1 if train_ds.get_labels() == None else len(
        train_ds.get_labels())

    # Step1: Initialize the origin BERT model.
    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_labels)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Step2: Convert origin model to supernet.
    sp_config = supernet(expand_ratio=args.width_mult_list)
    model = Convert(sp_config).convert(model)

    # Use weights saved in the dictionary to initialize supernet. 
    weights = os.path.join(args.model_name_or_path, 'model_state.pdparams')
    origin_weights = paddle.load(weights)
    model.set_state_dict(origin_weights)

    # Step3: Define teacher model.
    teacher_model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_labels)
    new_dict = utils.utils.remove_model_fn(teacher_model, origin_weights)
    teacher_model.set_state_dict(new_dict)
    del origin_weights, new_dict

    default_run_config = {'elastic_depth': args.depth_mult_list}
    run_config = RunConfig(**default_run_config)

    # Step4: Config about distillation.
    mapping_layers = ['bert.embeddings']
    for idx in range(model.bert.config['num_hidden_layers']):
        mapping_layers.append('bert.encoder.layers.{}'.format(idx))

    default_distill_config = {
        'lambda_distill': args.lambda_rep,
        'teacher_model': teacher_model,
        'mapping_layers': mapping_layers,
    }
    distill_config = DistillConfig(**default_distill_config)

    # Step5: Config in supernet training.
    ofa_model = OFA(model,
                    run_config=run_config,
                    distill_config=distill_config,
                    elastic_order=['depth'])
    #elastic_order=['width'])

    criterion = paddle.nn.loss.CrossEntropyLoss() if train_ds.get_labels(
    ) else paddle.nn.loss.MSELoss()

    metric = metric_class()
    teacher_metric = metric_class()

    if args.task_name == "mnli":
        dev_data_loader = (dev_data_loader_matched, dev_data_loader_mismatched)

    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.learning_rate,
        lambda current_step, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps if args.max_steps > 0 else
        (len(train_data_loader) * args.num_train_epochs): float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))))

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=ofa_model.model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in ofa_model.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        # Step6: Set current epoch and task.
        ofa_model.set_epoch(epoch)
        ofa_model.set_task('depth')

        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch

            for depth_mult in args.depth_mult_list:
                for width_mult in args.width_mult_list:
                    # Step7: Broadcast supernet config from width_mult,
                    # and use this config in supernet training.
                    net_config = utils.dynabert_config(ofa_model, width_mult,
                                                       depth_mult)
                    ofa_model.set_net_config(net_config)
                    logits, teacher_logits = ofa_model(
                        input_ids, segment_ids, attention_mask=[None, None])
                    rep_loss = ofa_model.calc_distill_loss()
                    if args.task_name == 'sts-b':
                        logit_loss = 0.0
                    else:
                        logit_loss = soft_cross_entropy(logits,
                                                        teacher_logits.detach())
                    loss = rep_loss + args.lambda_logit * logit_loss
                    loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ofa_model.model.clear_gradients()

            if global_step % args.logging_steps == 0:
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss,
                           args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                if args.task_name == "mnli":
                    evaluate(
                        teacher_model,
                        criterion,
                        metric,
                        teacher_metric,
                        dev_data_loader_matched,
                        width_mult=100)
                    evaluate(
                        teacher_model,
                        criterion,
                        metric,
                        teacher_metric,
                        dev_data_loader_mismatched,
                        width_mult=100)
                else:
                    evaluate(
                        teacher_model,
                        criterion,
                        metric,
                        teacher_metric,
                        dev_data_loader,
                        width_mult=100)
                for depth_mult in args.depth_mult_list:
                    for width_mult in args.width_mult_list:
                        net_config = utils.dynabert_config(
                            ofa_model, width_mult, depth_mult)
                        ofa_model.set_net_config(net_config)
                        tic_eval = time.time()
                        if args.task_name == "mnli":
                            acc = evaluate(
                                ofa_model, criterion, metric, teacher_metric,
                                dev_data_loader_matched, width_mult, depth_mult)
                            evaluate(ofa_model, criterion, metric,
                                     teacher_metric, dev_data_loader_mismatched,
                                     width_mult, depth_mult)
                            print("eval done total : %s s" %
                                  (time.time() - tic_eval))
                        else:
                            acc = evaluate(ofa_model, criterion, metric,
                                           teacher_metric, dev_data_loader,
                                           width_mult, depth_mult)
                            print("eval done total : %s s" %
                                  (time.time() - tic_eval))

                        if (not args.n_gpu > 1
                            ) or paddle.distributed.get_rank() == 0:
                            output_dir = os.path.join(args.output_dir,
                                                      "model_%d" % global_step)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # need better way to get inner model of DataParallel
                            model_to_save = model._layers if isinstance(
                                model, paddle.DataParallel) else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)

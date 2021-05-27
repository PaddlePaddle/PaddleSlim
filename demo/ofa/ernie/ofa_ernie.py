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

import os
import re
import time
import json
from random import random
from tqdm import tqdm
from functools import reduce, partial

import numpy as np
import math
import logging
import argparse

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig, utils

from propeller import log
import propeller.paddle as propeller

from ernie.modeling_ernie import ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import LinearDecay
from ernie_supernet.importance import compute_neuron_head_importance, reorder_neuron_head
from ernie_supernet.optimization import AdamW
from ernie_supernet.modeling_ernie_supernet import get_config
from paddleslim.nas.ofa.convert_super import Convert, supernet


def soft_cross_entropy(inp, target):
    inp_likelihood = L.log_softmax(inp, axis=-1)
    target_prob = L.softmax(target, axis=-1)
    return -1. * L.mean(L.reduce_sum(inp_likelihood * target_prob, dim=-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('classify model with ERNIE')
    parser.add_argument(
        '--from_pretrained',
        type=str,
        required=True,
        help='pretrained model directory or tag')
    parser.add_argument(
        '--max_seqlen',
        type=int,
        default=128,
        help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=32, help='batchsize')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='data directory includes train / develop data')
    parser.add_argument('--task', type=str, default='xnli', help='task name')
    parser.add_argument(
        '--use_lr_decay',
        action='store_true',
        help='if set, learning rate will decay to zero at `max_steps`')
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        default=0.1,
        help='if use_lr_decay is set, '
        'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`'
    )
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument(
        '--inference_model_dir',
        type=str,
        default='ofa_ernie_inf',
        help='inference model output directory')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='ofa_ernie_save',
        help='model output directory')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.01,
        help='weight decay, aka L2 regularizer')
    parser.add_argument(
        '--width_lambda1',
        type=float,
        default=1.0,
        help='scale for logit loss in elastic width')
    parser.add_argument(
        '--width_lambda2',
        type=float,
        default=0.1,
        help='scale for rep loss in elastic width')
    parser.add_argument(
        '--depth_lambda1',
        type=float,
        default=1.0,
        help='scale for logit loss in elastic depth')
    parser.add_argument(
        '--depth_lambda2',
        type=float,
        default=1.0,
        help='scale for rep loss in elastic depth')
    parser.add_argument(
        '--reorder_weight',
        action='store_false',
        help='Whether to reorder weight')
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help='checkpoint to warm start from')
    parser.add_argument(
        '--width_mult_list',
        nargs='+',
        type=float,
        default=[1.0, 0.75, 0.5, 0.25],
        help="width mult in compress")
    parser.add_argument(
        '--depth_mult_list',
        nargs='+',
        type=float,
        default=[1.0, 2 / 3],
        help="depth mult in compress")
    args = parser.parse_args()

    if args.task == 'sts-b':
        mode = 'regression'
    else:
        mode = 'classification'

    tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'seg_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.TextColumn(
            'seg_b',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.LabelColumn(
            'label',
            vocab_dict={
                b"contradictory": 0,
                b"contradiction": 0,
                b"entailment": 1,
                b"neutral": 2,
            }),
    ])

    def map_fn(seg_a, seg_b, label):
        seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
        return sentence, segments, label


    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0, 0))

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0, 0))

    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1])
    types = ('int64', 'int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types

    place = F.CUDAPlace(0)
    with FD.guard(place):
        model = ErnieModelForSequenceClassification.from_pretrained(
            args.from_pretrained, num_labels=3, name='')
        setattr(model, 'return_additional_info', True)

        origin_weights = {}
        for name, param in model.named_parameters():
            origin_weights[name] = param

        sp_config = supernet(expand_ratio=args.width_mult_list)
        model = Convert(sp_config).convert(model)
        utils.set_state_dict(model, origin_weights)
        del origin_weights

        teacher_model = ErnieModelForSequenceClassification.from_pretrained(
            args.from_pretrained, num_labels=3, name='teacher')
        setattr(teacher_model, 'return_additional_info', True)

        default_run_config = {
            'n_epochs': [[4 * args.epoch], [6 * args.epoch]],
            'init_learning_rate': [[args.lr], [args.lr]],
            'elastic_depth': args.depth_mult_list,
            'dynamic_batch_size': [[1, 1], [1, 1]]
        }
        run_config = RunConfig(**default_run_config)

        model_cfg = get_config(args.from_pretrained)

        default_distill_config = {'teacher_model': teacher_model}
        distill_config = DistillConfig(**default_distill_config)

        ofa_model = OFA(model,
                        run_config,
                        distill_config=distill_config,
                        elastic_order=['width', 'depth'])

        ### suppose elastic width first
        if args.reorder_weight:
            head_importance, neuron_importance = compute_neuron_head_importance(
                args, ofa_model.model, dev_ds, place, model_cfg)
            reorder_neuron_head(ofa_model.model, head_importance,
                                neuron_importance)
        #################

        if args.init_checkpoint is not None:
            log.info('loading checkpoint from %s' % args.init_checkpoint)
            sd, _ = FD.load_dygraph(args.init_checkpoint)
            ofa_model.model.set_dict(sd)

        g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
        if args.use_lr_decay:
            opt = AdamW(
                learning_rate=LinearDecay(args.lr,
                                          int(args.warmup_proportion *
                                              args.max_steps), args.max_steps),
                parameter_list=ofa_model.model.parameters(),
                weight_decay=args.wd,
                grad_clip=g_clip)
        else:
            opt = AdamW(
                args.lr,
                parameter_list=ofa_model.model.parameters(),
                weight_decay=args.wd,
                grad_clip=g_clip)

        for epoch in range(max(run_config.n_epochs[-1])):
            ofa_model.set_epoch(epoch)
            if epoch <= int(max(run_config.n_epochs[0])):
                ofa_model.set_task('width')
                depth_mult_list = [1.0]
            else:
                ofa_model.set_task('depth')
                depth_mult_list = run_config.elastic_depth
            for step, d in enumerate(
                    tqdm(
                        train_ds.start(place), desc='training')):
                ids, sids, label = d

                accumulate_gradients = dict()
                for param in opt._parameter_list:
                    accumulate_gradients[param.name] = 0.0

                for depth_mult in depth_mult_list:
                    for width_mult in args.width_mult_list:
                        net_config = utils.dynabert_config(
                            ofa_model, width_mult, depth_mult=depth_mult)
                        ofa_model.set_net_config(net_config)

                        student_output, teacher_output = ofa_model(
                            ids,
                            sids,
                            labels=label,
                            num_layers=model_cfg['num_hidden_layers'])
                        loss, student_logit, student_reps = student_output[
                            0], student_output[1], student_output[2]['hiddens']
                        teacher_logit, teacher_reps = teacher_output[
                            1], teacher_output[2]['hiddens']

                        if ofa_model.task == 'depth':
                            depth_mult = ofa_model.current_config['depth']
                            depth = round(model_cfg['num_hidden_layers'] *
                                          depth_mult)
                            kept_layers_index = []
                            for i in range(1, depth + 1):
                                kept_layers_index.append(
                                    math.floor(i / depth_mult) - 1)

                            if mode == 'classification':
                                logit_loss = soft_cross_entropy(
                                    student_logit, teacher_logit.detach())
                            else:
                                logit_loss = 0.0

                            ### hidden_states distillation loss
                            rep_loss = 0.0
                            for stu_rep, tea_rep in zip(
                                    student_reps,
                                    list(teacher_reps[i]
                                         for i in kept_layers_index)):
                                tmp_loss = L.mse_loss(stu_rep, tea_rep.detach())
                                rep_loss += tmp_loss

                            loss = args.width_lambda1 * logit_loss + args.width_lambda2 * rep_loss

                        else:
                            ### logit distillation loss
                            if mode == 'classification':
                                logit_loss = soft_cross_entropy(
                                    student_logit, teacher_logit.detach())
                            else:
                                logit_loss = 0.0

                            ### hidden_states distillation loss
                            rep_loss = 0.0
                            for stu_rep, tea_rep in zip(student_reps,
                                                        teacher_reps):
                                tmp_loss = L.mse_loss(stu_rep, tea_rep.detach())
                                rep_loss += tmp_loss

                            loss = args.width_lambda1 * logit_loss + args.width_lambda2 * rep_loss

                        if step % 10 == 0:
                            print('train loss %.5f lr %.3e' %
                                  (loss.numpy(), opt.current_step_lr()))

                        loss.backward()
                        param_grads = opt.backward(loss)
                        for param in opt._parameter_list:
                            accumulate_gradients[param.name] += param.gradient()
                for k, v in param_grads:
                    assert k.name in accumulate_gradients.keys(
                    ), "{} not in accumulate_gradients".format(k.name)
                    v.set_value(accumulate_gradients[k.name])
                opt.apply_optimize(
                    loss, startup_program=None, params_grads=param_grads)
                ofa_model.model.clear_gradients()

                if step % 100 == 0:
                    for depth_mult in depth_mult_list:
                        for width_mult in args.width_mult_list:
                            net_config = utils.dynabert_config(
                                ofa_model, width_mult, depth_mult=depth_mult)
                            ofa_model.set_net_config(net_config)

                            acc = []
                            tea_acc = []
                            with FD.base._switch_tracer_mode_guard_(
                                    is_train=False):
                                ofa_model.model.eval()
                                for step, d in enumerate(
                                        tqdm(
                                            dev_ds.start(place),
                                            desc='evaluating %d' % epoch)):
                                    ids, sids, label = d
                                    [loss, logits,
                                     _], [_, tea_logits, _] = ofa_model(
                                         ids,
                                         sids,
                                         labels=label,
                                         num_layers=model_cfg[
                                             'num_hidden_layers'])
                                    a = L.argmax(logits, -1) == label
                                    acc.append(a.numpy())

                                    ta = L.argmax(tea_logits, -1) == label
                                    tea_acc.append(ta.numpy())
                                ofa_model.model.train()
                            print(
                                'width_mult: %f, depth_mult: %f: acc %.5f, teacher acc %.5f'
                                % (width_mult, depth_mult,
                                   np.concatenate(acc).mean(),
                                   np.concatenate(tea_acc).mean()))
        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            F.save_dygraph(ofa_model.model.state_dict(), args.save_dir)

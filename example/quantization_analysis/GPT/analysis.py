# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random
import numpy as np
import argparse
import time

import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.quant.analysis_qat import AnalysisQAT
from ppfleetx.data import build_dataloader
from ppfleetx.distributed.apis import env
from utils import parse_config


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='analysis_results',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser


def eval_reader_wrapper(reader):
    def gen():
        for data in reader:
            tokens, loss_mask, attention_mask, position_ids, labels, info = data
            in_dict = {}
            in_dict['tokens'] = tokens
            in_dict['ids'] = position_ids
            yield in_dict, labels, loss_mask, info

    return gen


def eval_function(exe, program, feed_names, fetch_list):
    tic_eval = time.time()
    score_name = "loss" if not global_config['cloze_eval'] else "number correct"
    first_step = True
    eval_losses = []
    total_score = 0
    for eval_step, (data, labels, loss_mask, info) in enumerate(eval_loader()):
        preds = exe.run(program=program,
                        feed=data,
                        fetch_list=fetch_list,
                        return_numpy=False)

        paddle.disable_static()

        labels = paddle.to_tensor(labels)
        preds = paddle.to_tensor(preds[0])
        loss_mask = paddle.to_tensor(loss_mask)
        info = paddle.to_tensor(info)

        if not global_config['cloze_eval']:
            if first_step:
                num_original_tokens = info.numpy()[0][0]
                num_tokenized_tokens = info.numpy()[0][1]
                first_step = False

            masked_lm_loss = paddle.nn.functional.cross_entropy(
                preds, labels, reduction="none")
            loss = paddle.sum(masked_lm_loss * loss_mask)
            eval_losses.append(loss.numpy()[0])
            total_score += loss.numpy() / (num_tokenized_tokens - 1)

        else:
            if first_step:
                num_examples = info.numpy()[0][0]
                first_step = False
            outputs = paddle.argmax(preds, -1)
            acc = paddle.cast(outputs == labels, 'float32')
            acc = paddle.where(
                paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
            acc = paddle.sum(paddle.prod(acc, -1))
            eval_losses.append(acc.numpy()[0])
            total_score += acc.numpy()[0]

        if eval_step != 0 and (eval_step % 10 == 0):
            print("[eval] step: %d, batch: %d, %s: %.9f, speed: %.2f step/s" %
                  (eval_step, eval_step, score_name, total_score,
                   1. / (time.time() - tic_eval)))
            tic_eval = time.time()
        paddle.enable_static()

    metric = None
    if not global_config['cloze_eval']:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = ' validation results on {} | '.format(gpt_config['Data'][
            'Eval']['dataset']['name'])
        string += 'avg loss: {:.4E} | '.format(total_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
        metric = ppl
    else:
        num_correct = float(total_score)
        acc = float(num_correct / num_examples)
        string = ' validation results on {} | '.format(gpt_config['Data'][
            'Eval']['dataset']['name'])
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(num_examples)
        string += 'avg accuracy: {:.4E}'.format(acc)
        metric = acc

    print(string)
    return metric


def main():
    global global_config, all_config
    all_config = load_slim_config(FLAGS.config_path)
    assert "Global" in all_config, "Key 'Global' not found in config file. \n{}".format(
        all_config)
    global_config = all_config["Global"]

    seed = all_config['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    env.set_seed(seed)

    global gpt_config
    gpt_config = parse_config(global_config['reader_config'])

    if not global_config['cloze_eval']:
        gpt_config['Data']['Eval']['dataset']['name'] = "LM_Eval_Dataset"
    else:
        gpt_config['Data']['Eval']['dataset']['name'] = "Lambada_Eval_Dataset"

    valid_data_loader = build_dataloader(gpt_config['Data'], "Eval")

    global eval_loader
    eval_loader = eval_reader_wrapper(valid_data_loader)

    analyzer = AnalysisQAT(
        quant_model_dir=global_config["quant_model_dir"],
        float_model_dir=global_config["float_model_dir"],
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"],
        quantizable_op_type=global_config['quantizable_op_type'],
        qat_metric=global_config['qat_metric']
        if 'qat_metric' in global_config else None,
        eval_function=eval_function,
        data_loader=eval_loader,
        save_dir=FLAGS.save_dir,
        resume=global_config['resume'], )
    analyzer.metric_error_analyse()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()

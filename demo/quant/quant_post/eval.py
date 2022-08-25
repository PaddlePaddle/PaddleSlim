#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import numpy as np
import argparse
import functools

import paddle
sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
sys.path[1] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import imagenet_reader as reader
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('model_path', str,  "./pruning/checkpoints/resnet50/2/eval_model/",                 "Whether to use pretrained model.")
add_arg('model_name', str,  'model.pdmodel', "model filename for inference model")
add_arg('params_name', str, 'model.pdiparams', "params filename for inference model")
add_arg('batch_size',       int,  64,                 "Minibatch size.")
# yapf: enable


def eval(args):
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    val_program, feed_target_names, fetch_targets = paddle.fluid.io.load_inference_model(
        args.model_path,
        exe,
        model_filename=args.model_name,
        params_filename=args.params_name)
    val_dataset = reader.ImageNetDataset(mode='val')

    image = paddle.static.data(
        name='image', shape=[None, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    val_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=True,
        batch_size=args.batch_size,
        use_shared_memory=True,
        shuffle=False)

    results = []
    for batch_id, data in enumerate(val_loader()):
        # top1_acc, top5_acc
        if len(feed_target_names) == 1:
            # eval "infer model", which input is image, output is classification probability
            image = data[0]
            label = data[1]
            pred = exe.run(val_program,
                           feed={feed_target_names[0]: image},
                           fetch_list=fetch_targets)
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
            image = data[0]
            label = data[1]
            result = exe.run(val_program,
                             feed={
                                 feed_target_names[0]: image,
                                 feed_target_names[1]: label
                             },
                             fetch_list=fetch_targets)
            result = [np.mean(r) for r in result]
            results.append(result)
        if batch_id % 100 == 0:
            print('Eval iter: ', batch_id)
    result = np.mean(np.array(results), axis=0)
    print("top1_acc/top5_acc= {}".format(result))
    sys.stdout.flush()


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()

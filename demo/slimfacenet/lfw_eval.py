# ================================================================
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import argparse
import time
import scipy.io
import numpy as np

import paddle
from paddle import fluid

from dataloader.casia import CASIA_Face
from dataloader.lfw import LFW
from paddleslim import models


def parse_filelist(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = 'lfw-112X96'
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0],
                                 p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0],
                                 p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0],
                                 p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2],
                                 p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    return [nameLs, nameRs, folds, flags]


def get_accuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def get_threshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = get_accuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(root='result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(
            np.concatenate(
                (featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(
            np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(
            np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = get_threshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = get_accuracy(scores[testFold[0]], flags[testFold[0]],
                               threshold)
    return ACCs


def test(test_reader, flods, flags, net, args):
    net.eval()
    featureLs = None
    featureRs = None
    for idx, data in enumerate(test_reader()):
        data_list = [[] for _ in range(4)]
        for _ in range(len(data)):
            data_list[0].append(data[_][0])
            data_list[1].append(data[_][1])
            data_list[2].append(data[_][2])
            data_list[3].append(data[_][3])
        res = [
            net(fluid.dygraph.to_variable(np.array(d))).numpy()
            for d in data_list
        ]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(args.feature_save_dir, result)
    ACCs = evaluation_10_fold(args.feature_save_dir)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PaddlePaddle SlimFaceNet')
    parser.add_argument(
        '--use_gpu', default=0, type=int, help='Use GPU or not, 0 is not used')
    parser.add_argument(
        '--test_data_dir', default='./lfw', type=str, help='lfw_data_dir')
    parser.add_argument(
        '--resume', default='output/0', type=str, help='resume')
    parser.add_argument(
        '--feature_save_dir',
        default='result.mat',
        type=str,
        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()

    place = fluid.CPUPlace() if args.use_gpu == 0 else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        train_dataset = CASIA_Face(root=args.train_data_dir)
        nl, nr, flods, flags = parse_filelist(args.test_data_dir)
        test_dataset = LFW(nl, nr)
        test_reader = paddle.fluid.io.batch(
            test_dataset.reader,
            batch_size=args.test_batchsize,
            drop_last=False)

        net = models.__dict__[args.model](class_dim=train_dataset.class_nums)
        if args.resume:
            assert os.path.exists(args.resume + ".pdparams"
                                  ), "Given dir {}.pdparams not exist.".format(
                                      args.resume)
            para_dict, opti_dict = fluid.dygraph.load_dygraph(args.resume)
            net.set_dict(para_dict)

        test(test_reader, flods, flags, net, args)

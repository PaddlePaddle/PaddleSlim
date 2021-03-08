#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import argparse
import ast

import numpy as np
import paddle
from dataset.data_loader import create_eval_data
from metric.inception import InceptionV3
from metric import get_fid
from model.test_model import TestModel
from utils import util


def main(cfgs):
    if cfgs.config_str is not None:
        assert 'super' in cfgs.netG or 'sub' in cfgs.netG
        config = decode_config(cfgs.config_str)
    else:
        assert 'super' not in cfgs.model
        config = None

    data_loader, id2name = create_eval_data(cfgs, direction=cfgs.direction)
    model = TestModel(cfgs)
    model.setup()  ### load_network

    fakes, names = [], []
    for i, data in enumerate(data_loader()):
        model.set_input(data)
        if i == 0 and cfgs.need_profile:
            flops, params = model.profile(config)
            print('FLOPs: %.3fG, params: %.3fM' % (flops / 1e9, params / 1e6))
            sys.exit(0)
        model.test(config)
        generated = model.fake_B
        fakes.append(generated.detach().numpy())
        name = id2name[i]
        print(name)
        save_path = os.path.join(cfgs.save_dir, 'test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, name)
        names.append(name)
        if i < cfgs.num_test:
            image = util.tensor2img(generated)
            util.save_image(image, save_path)

    paddle.enable_static()
    if not cfgs.no_fid:
        print('Calculating FID...')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx])
        npz = np.load(cfgs.real_stat_path)
        fid = get_fid(fakes, inception_model, npz, cfgs.inception_model_path)
        print('fid score: %#.2f' % fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='Whether to use GPU in train/test model.')
    parser.add_argument(
        '--need_profile',
        type=ast.literal_eval,
        default=True,
        help='Whether to profile model.')
    parser.add_argument(
        '--no_fid',
        type=ast.literal_eval,
        default=False,
        help='Whether to get fid.')

    parser.add_argument(
        '--batch_size', type=int, default=1, help="Minbatch size")
    parser.add_argument(
        '--shuffle',
        type=ast.literal_eval,
        default=False,
        help="Whether to shuffle data")
    parser.add_argument(
        '--dataset',
        type=str,
        default='horse2zebra',
        help="The name of dataset")
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data',
        help="The dictionary of data")
    parser.add_argument(
        '--model', type=str, default='cycle_gan', help="model name")
    parser.add_argument(
        '--image_size',
        type=int,
        default=286,
        help="The image size when load image")
    parser.add_argument(
        '--crop_size',
        type=int,
        default=256,
        help="The crop size used to crop image")
    parser.add_argument(
        '--num_test',
        type=int,
        default=np.inf,
        help="the number of fake images to save")

    parser.add_argument(
        '--real_stat_path',
        type=str,
        default='real_stat/horse2zebra_B.npz',
        help="path of real stat")
    parser.add_argument(
        '--ngf', type=int, default=None, help="Base channels in generator")
    parser.add_argument(
        '--netG',
        type=str,
        default='mobile_resnet_9blocks',
        help="Which generator network to choose")
    parser.add_argument(
        '--dropout_rate', type=float, default=0, help="dropout rate")
    parser.add_argument(
        '--restore_G_path',
        type=str,
        default=None,
        help="the pretrain model path of generator")

    parser.add_argument(
        '--input_nc', type=int, default=3, help="input channel")
    parser.add_argument(
        '--output_nc', type=int, default=3, help="output channel")
    parser.add_argument(
        '--norm_type',
        type=str,
        default='instance',
        help="The type of normalization")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./test_mobile_pth',
        help="The directory the model and the test result to save")
    parser.add_argument(
        '--inception_model_path',
        type=str,
        default='metric/params_inceptionV3',
        help="the directory of inception model, used in computing fid")
    parser.add_argument(
        '--direction', type=str, default='AtoB', help="direction of generator")

    parser.add_argument(
        '--config_set',
        type=str,
        default=None,
        help="a set of configuration to get subnets of supernet")
    parser.add_argument(
        '--config_str',
        type=str,
        default=None,
        help="the configuration string used to get specific subnet of supernet")

    cfgs = parser.parse_args()
    main(cfgs)

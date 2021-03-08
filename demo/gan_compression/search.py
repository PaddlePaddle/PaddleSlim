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
import pickle
import random
import argparse
import ast
import sys
import time

import numpy as np
import paddle

from configs import encode_config
from data_loader import create_eval_data
from metric.inception import InceptionV3
from metric import get_fid

from model.test_model import TestModel
from utils import util


def main(cfgs):
    if 'resnet' in cfgs.netG:
        from configs.resnet_configs import get_configs
    else:
        raise NotImplementedError
    configs = get_configs(config_name=cfgs.config_set)
    configs = list(configs.all_configs())

    data_loader, id2name = create_eval_data(cfgs, direction=cfgs.direction)
    model = TestModel(cfgs)
    model.setup()  ### load_network

    ### this input used in compute model flops and params
    for data in data_loader:
        model.set_input(data)
        break

    npz = np.load(cfgs.real_stat_path)
    results = []
    for config in configs:
        fakes, names = [], []
        flops, _ = model.profile(config=config)
        s_time = time.time()
        for i, data in enumerate(data_loader()):
            model.set_input(data)
            model.test(config)
            generated = model.fake_B
            fakes.append(generated.detach().numpy())
            name = id2name[i]
            save_path = os.path.join(cfgs.save_dir, 'test' + str(config))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, name)
            names.append(name)
            if i < cfgs.num_test:
                image = util.tensor2img(generated)
                util.save_image(image, save_path)

        result = {
            'config_str': encode_config(config),
            'flops': flops
        }  ### compute FLOPs

        paddle.enable_static()
        if not cfgs.no_fid:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception_model = InceptionV3([block_idx])
            fid = get_fid(
                fakes,
                inception_model,
                npz,
                cfgs.inception_model_path,
                batch_size=cfgs.batch_size,
                use_gpu=cfgs.use_gpu)
            result['fid'] = fid
        paddle.disable_static()

        e_time = (time.time() - s_time) / 60
        result['time'] = e_time
        print(result)
        results.append(result)

    if not os.path.exists(cfgs.save_dir):
        os.makedirs(os.path.dirname(cfgs.save_dir))
    save_file = os.path.join(cfgs.save_dir, 'search_result.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    print('Successfully finish searching!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='Whether to use GPU in train/test model.')
    parser.add_argument(
        '--no_fid',
        type=ast.literal_eval,
        default=False,
        help='Whether to get fid.')

    parser.add_argument(
        '--batch_size', type=int, default=32, help="Minbatch size")
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
        help="")
    parser.add_argument(
        '--ngf', type=int, default=32, help="Base channels in generator")
    parser.add_argument(
        '--netG',
        type=str,
        default='super_mobile_resnet_9blocks',
        help="Which generator network to choose")
    parser.add_argument(
        '--dropout_rate', type=float, default=0, help="dropout rate")
    parser.add_argument(
        '--restore_G_path',
        type=str,
        default='./output/supernet/last_stu_netG',
        help="the pretrain model path of  generator")

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
        default='./search_result',
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
        default='channels-32',
        help="a set of configuration to get subnets of supernet")
    parser.add_argument(
        '--config_str',
        type=str,
        default=None,
        help="the configuration string used to get specific subnet of supernet")

    cfgs = parser.parse_args()
    main(cfgs)

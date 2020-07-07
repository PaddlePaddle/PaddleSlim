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
import argparse
import ast

import models
import distillers
import supernets

__all__ = ['configs']


class configs:
    def base_config(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            default='cycle_gan',
            help="The model want to compression")
        parser.add_argument(
            '--task',
            type=str,
            default='mobile+distiller+supernet',
            help="Base channels in generator")
        parser.add_argument(
            '--distiller',
            type=str,
            default='resnet',
            help="generator network in distiller")
        parser.add_argument(
            '--supernet',
            type=str,
            default='resnet',
            help="generator network in supernet")
        parser.add_argument(
            '--gpu_num', type=int, default='1', help='GPU number.')
        ### data
        parser.add_argument(
            '--batch_size', type=int, default=1, help="Minbatch size")
        parser.add_argument(
            '--shuffle',
            type=ast.literal_eval,
            default=True,
            help="Whether to shuffle data")
        parser.add_argument(
            '--flip',
            type=ast.literal_eval,
            default=True,
            help="Whether to flip data randomly")
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
            '--crop_type',
            type=str,
            default='Random',
            help="Which method to crop image")
        ##############
        parser.add_argument(
            '--gan_loss_mode',
            type=str,
            default='lsgan',
            help="The mode used to compute gan loss")
        parser.add_argument(
            '--ngf', type=int, default=64, help="Base channels in generator")
        parser.add_argument(
            '--netG',
            type=str,
            default='mobile_resnet_9blocks',
            help="Which generator network to choose")
        parser.add_argument(
            '--dropout_rate',
            type=float,
            default=0,
            help="dropout rate in generator")
        ###############
        parser.add_argument(
            '--input_nc', type=int, default=3, help="Channel of input")
        parser.add_argument(
            '--output_nc', type=int, default=3, help="Channel of output")
        parser.add_argument(
            '--norm_type',
            type=str,
            default='instance',
            help="The type of normalization")
        parser.add_argument(
            '--save_dir',
            type=str,
            default='./output',
            help="The directory the model and the test result to save")
        parser.add_argument(
            '--netD',
            type=str,
            default='n_layers',
            help="Which discriminator network to choose")
        parser.add_argument(
            '--ndf',
            type=int,
            default=64,
            help="Base channels in discriminator")
        parser.add_argument(
            '--n_layer_D',
            type=int,
            default=3,
            help="The number of layer in discriminator, only used when netD == n_layers"
        )
        parser.add_argument(
            '--beta1', type=float, default=0.5, help="momentum term of adam")
        parser.add_argument(
            '--direction',
            type=str,
            default='AtoB',
            help="the direction of generator")
        parser.add_argument(
            '--step_per_epoch',
            type=int,
            default=1333,
            help="The number of step in each epoch")
        parser.add_argument(
            '--inception_model',
            type=str,
            default='metric/params_inceptionV3',
            help="The directory of inception model, used in computing fid")

        parser.add_argument(
            '--restore_D_path',
            type=str,
            default=None,
            help="the pretrain model path of discriminator")
        parser.add_argument(
            '--restore_G_path',
            type=str,
            default=None,
            help="the pretrain model path of generator")
        parser.add_argument(
            '--restore_A_path',
            type=str,
            default=None,
            help="the pretrain model path of list of conv used in distiller")
        parser.add_argument(
            '--restore_O_path',
            type=str,
            default=None,
            help="the pretrain model path of optimization")

        parser.add_argument(
            '--print_freq', type=int, default=1, help="print log frequency")
        parser.add_argument(
            '--save_freq',
            type=int,
            default=1,
            help="the epoch frequency to save model")

        return parser

    def get_all_config(self):
        parser = argparse.ArgumentParser(description="Configs for all model")
        parser = self.base_config(parser)
        cfg, _ = parser.parse_known_args()

        task = cfg.task
        tasks = task.split('+')
        load_resbase = True
        for t in tasks:
            if t == 'mobile':
                model = cfg.model
                model_parser = models.get_special_cfg(model)
                parser = model_parser(parser)
            elif t == 'distiller':
                model = cfg.distiller
                model_parser = distillers.get_special_cfg(model)
                parser = model_parser(parser, load_resbase)
                load_resbase = False
            elif t == 'supernet':
                model = cfg.supernet
                model_parser = supernets.get_special_cfg(model)
                parser = model_parser(parser, load_resbase)
                load_resbase = False
            else:
                raise NotImplementedError(
                    "task name {} is error, please check it".format(t))
        self.parser = parser
        return parser.parse_args()

    def print_configs(self, cfgs):
        print("-----------  Configuration Arguments -----------")
        for arg, value in sorted(vars(cfgs).items()):
            print("%s: %s" % (arg, value))
        print("------------------------------------------------")

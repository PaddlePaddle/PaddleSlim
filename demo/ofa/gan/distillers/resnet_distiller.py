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
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D
from distillers.base_resnet_distiller import BaseResnetDistiller
from utils import util
from utils.weight_transfer import load_pretrained_weight
from models import loss
from metric import get_fid


class ResnetDistiller(BaseResnetDistiller):
    @staticmethod
    def add_special_cfgs(parser, load_pre=False):
        parser.add_argument(
            '--distiller_lr',
            type=float,
            default=2e-4,
            help="Initial learning rate in train distiller net")
        parser.add_argument(
            '--distiller_epoch',
            type=int,
            default=200,
            help="The number of epoch to train distiller net")
        parser.add_argument(
            '--distiller_nepochs',
            type=int,
            default=100,
            help="number of epochs with the initial learning rate")
        parser.add_argument(
            '--distiller_nepochs_decay',
            type=int,
            default=100,
            help="number of epochs to linearly decay learning rate to zero")
        parser.add_argument(
            '--distiller_scheduler',
            type=str,
            default='linear',
            help="learning rate scheduler in train distiller net")
        parser.add_argument(
            '--distiller_student_netG',
            type=str,
            default='mobile_resnet_9blocks',
            help="Which student generator network to choose in distiller")
        parser.add_argument(
            '--pretrained_ngf',
            type=int,
            default=64,
            help="Base channels in generator")
        parser.add_argument(
            '--pretrained_netG',
            type=str,
            default='mobile_resnet_9blocks',
            help="Which generator network to choose in pretrain model")
        parser.add_argument(
            '--restore_pretrained_G_path',
            type=str,
            default=None,
            help="the pretrain model of pretrain_model used in distiller")
        if load_pre:
            super(ResnetDistiller, ResnetDistiller).add_special_cfgs(parser)
        return parser

    def __init__(self, cfgs):
        super(ResnetDistiller, self).__init__(cfgs, task='distiller')
        self.best_fid = 1e9
        self.fids = []
        self.npz = np.load(cfgs.real_stat_path)

    def forward(self):
        with paddle.no_grad():
            self.Tfake_B = self.netG_teacher(self.real_A)
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, Conv2D)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Tact.stop_gradient = True
            Sact = self.Sacts[n]
            ### 1x1 conv to match channels
            Sact = netA(Sact)
            loss = F.mse_loss(Sact, Tact)
            setattr(self, 'loss_G_distill%d' % i, loss)
            losses.append(loss)
        return sum(losses)

    def backward_G(self):
        self.loss_G_recon = loss.recon_loss(
            self.cfgs.recon_loss_mode, self.Sfake_B,
            self.Tfake_B) * self.cfgs.lambda_recon
        pred_fake = self.netD(self.Sfake_B)
        self.loss_G_gan = loss.gan_loss(
            self.cfgs.gan_loss_mode, pred_fake, True,
            for_discriminator=False) * self.cfgs.lambda_gan
        if self.cfgs.lambda_distill > 0:
            self.loss_G_distill = self.calc_distill_loss(
            ) * self.cfgs.lambda_distill
        else:
            self.loss_G_distill = 0

        self.loss_G = self.loss_G_gan + self.loss_G_recon + self.loss_G_distill
        self.loss_G.backward()

    def optimize_parameter(self):
        self.forward()

        self.set_stop_gradient(self.netD, False)
        self.backward_D()

        self.set_stop_gradient(self.netD, True)
        self.backward_G()
        self.optimizer_D.optimizer.minimize(self.loss_D)
        self.optimizer_D.optimizer.clear_gradients()
        self.optimizer_G.optimizer.minimize(self.loss_G)
        self.optimizer_G.optimizer.clear_gradients()

    def load_networks(self, model_weight=None):
        if self.cfgs.restore_pretrained_G_path.lower() != 'false':
            if self.cfgs.restore_pretrained_G_path != None:
                pretrained_G_path = self.cfgs.restore_pretrained_G_path
                util.load_network(self.netG_pretrained, pretrained_G_path)
            else:
                assert len(
                    model_weight
                ) != 0, "restore_pretrained_G_path and model_weight can not be None at the same time, if you donnot want to load pretrained model, please set restore_pretrained_G_path=Fasle"
                if self.cfgs.direction == 'AtoB':
                    self.netG_pretrained.set_dict(model_weight['netG_A'])
                else:
                    self.netG_pretrained.set_dict(model_weight['netG_B'])

            load_pretrained_weight(
                self.cfgs.pretrained_netG, self.cfgs.distiller_student_netG,
                self.netG_pretrained, self.netG_student,
                self.cfgs.pretrained_ngf, self.cfgs.student_ngf)
            del self.netG_pretrained

        super(ResnetDistiller, self).load_networks(model_weight)

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.cfgs.save_dir, 'distiller', 'eval',
                                str(step))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.netG_student.eval()
        fakes = []
        cnt = 0
        for i, data_i in enumerate(self.eval_dataloader):
            self.set_single_input(data_i)
            self.test()
            fakes.append(self.Sfake_B.detach().numpy())
            for j in range(len(self.Sfake_B)):
                if cnt < 10:
                    Sname = 'Sfake_' + str(i + j) + '.png'
                    Tname = 'Tfake_' + str(i + j) + '.png'
                    Sfake_im = util.tensor2img(self.Sfake_B[j])
                    Tfake_im = util.tensor2img(self.Tfake_B[j])
                    util.save_image(Sfake_im, os.path.join(save_dir, Sname))
                    util.save_image(Tfake_im, os.path.join(save_dir, Tname))
                cnt += 1

        suffix = self.cfgs.direction
        paddle.enable_static()
        fid = get_fid(fakes, self.inception_model, self.npz,
                      self.cfgs.inception_model)
        paddle.disable_static()
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        print("fid score is: %f, best fid score is %f" % (fid, self.best_fid))
        self.fids.append(fid)
        if len(self.fids) > 3:
            self.fids.pop(0)
        ret['metric/fid'] = fid
        ret['metric/fid-mean'] = sum(self.fids) / len(self.fids)
        ret['metric/fid-best'] = self.best_fid

        self.netG_student.train()
        return ret

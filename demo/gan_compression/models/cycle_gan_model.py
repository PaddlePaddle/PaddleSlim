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
from dataset.data_loader import create_eval_data
from utils.image_pool import ImagePool
from models import network, loss
from models.base_model import BaseModel
from metric.inception import InceptionV3
from utils import util, optimization
from metric import get_fid


class CycleGAN(BaseModel):
    @staticmethod
    def add_special_cfgs(parser):
        parser.add_argument(
            '--mobile_lr',
            type=float,
            default=2e-4,
            help="initial learning rate to train cyclegan")
        parser.add_argument(
            '--mobile_epoch',
            type=int,
            default=200,
            help="The number of epoch to train mobile net")
        parser.add_argument(
            '--mobile_nepochs',
            type=int,
            default=100,
            help="number of epochs with the initial learning rate")
        parser.add_argument(
            '--mobile_nepochs_decay',
            type=int,
            default=100,
            help="number of epochs to linearly decay learning rate to zero")
        parser.add_argument(
            '--mobile_scheduler',
            type=str,
            default='linear',
            help="learning rate scheduler")
        parser.add_argument(
            '--real_stat_A_path',
            type=str,
            default='real_stat/horse2zebra_A.npz',
            help="")
        parser.add_argument(
            '--real_stat_B_path',
            type=str,
            default='real_stat/horse2zebra_B.npz',
            help="")
        parser.add_argument(
            '--recon_loss_mode',
            type=str,
            default='l1',
            choices=['l1', 'l2'],
            help="")
        parser.add_argument(
            '--lambda_A',
            type=float,
            default=10.0,
            help="weight to scale cycle loss (A->B->A)")
        parser.add_argument(
            '--lambda_B',
            type=float,
            default=10.0,
            help="weight to scale cycle loss (B->A->B)")
        parser.add_argument(
            '--lambda_identity',
            type=float,
            default=0.5,
            help="Weight to scale identity mapping loss")
        parser.add_argument(
            '--pool_size', type=int, default=50, help="pool size in cyclegan")
        return parser

    def __init__(self, cfgs):
        super(CycleGAN, self).__init__()
        assert cfgs.direction == 'AtoB'
        self.cfgs = cfgs
        self.loss_names = [
            'D_A', 'G_A', 'G_cycle_A', 'G_idt_A', 'D_B', 'G_B', 'G_cycle_B',
            'G_idt_B'
        ]
        self.model_names = ['netG_A', 'netG_B', 'netD_A', 'netD_B']

        self.netG_A = network.define_G(cfgs.input_nc, cfgs.output_nc, cfgs.ngf,
                                       cfgs.netG, cfgs.norm_type,
                                       cfgs.dropout_rate)
        self.netG_B = network.define_G(cfgs.output_nc, cfgs.input_nc, cfgs.ngf,
                                       cfgs.netG, cfgs.norm_type,
                                       cfgs.dropout_rate)
        self.netD_A = network.define_D(cfgs.output_nc, cfgs.ndf, cfgs.netD,
                                       cfgs.norm_type, cfgs.n_layer_D)
        self.netD_B = network.define_D(cfgs.input_nc, cfgs.ndf, cfgs.netD,
                                       cfgs.norm_type, cfgs.n_layer_D)

        if self.cfgs.use_parallel:
            self.netG_A = paddle.DataParallel(self.netG_A, self.cfgs.strategy)
            self.netG_B = paddle.DataParallel(self.netG_B, self.cfgs.strategy)
            self.netD_A = paddle.DataParallel(self.netD_A, self.cfgs.strategy)
            self.netD_B = paddle.DataParallel(self.netD_B, self.cfgs.strategy)

        if cfgs.lambda_identity > 0.0:
            assert (cfgs.input_nc == cfgs.output_nc)
        self.fake_A_pool = ImagePool(cfgs.pool_size)
        self.fake_B_pool = ImagePool(cfgs.pool_size)

        self.optimizer_G = optimization.Optimizer(
            cfgs.mobile_lr,
            cfgs.mobile_scheduler,
            cfgs.step_per_epoch,
            cfgs.mobile_nepochs,
            cfgs.mobile_nepochs_decay,
            cfgs,
            parameter_list=(
                self.netG_A.parameters() + self.netG_B.parameters()))
        self.optimizer_D_A = optimization.Optimizer(
            cfgs.mobile_lr,
            cfgs.mobile_scheduler,
            cfgs.step_per_epoch,
            cfgs.mobile_nepochs,
            cfgs.mobile_nepochs_decay,
            cfgs,
            parameter_list=self.netD_A.parameters())
        self.optimizer_D_B = optimization.Optimizer(
            cfgs.mobile_lr,
            cfgs.mobile_scheduler,
            cfgs.step_per_epoch,
            cfgs.mobile_nepochs,
            cfgs.mobile_nepochs_decay,
            cfgs,
            parameter_list=self.netD_B.parameters())

        self.eval_dataloader_AtoB, self.name_AtoB = create_eval_data(
            cfgs, direction='AtoB')
        self.eval_dataloader_BtoA, self.name_BtoA = create_eval_data(
            cfgs, direction='BtoA')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])

        self.best_fid_A, self.best_fid_B = 1e9, 1e9
        self.fids_A, self.fids_B = [], []
        self.is_best = False
        self.npz_A = np.load(cfgs.real_stat_A_path)
        self.npz_B = np.load(cfgs.real_stat_B_path)

    def set_input(self, inputs):
        self.real_A = inputs[0] if self.cfgs.direction == 'AtoB' else inputs[1]
        self.real_B = inputs[1] if self.cfgs.direction == 'AtoB' else inputs[0]

    def set_single_input(self, inputs):
        self.real_A = inputs[0]

    def setup(self, model_weight=None):
        self.load_network()

    def load_network(self):
        for name in self.model_names:
            net = getattr(self, name, None)
            path = getattr(self.cfgs, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path)

    def save_network(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s' % (epoch, name)
                save_path = os.path.join(self.cfgs.save_dir, 'mobile',
                                         save_filename)
                net = getattr(self, name)
                paddle.save(net.state_dict(), save_path)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  ## G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  ## G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  ## G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  ## G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        ### real
        pred_real = netD(real)
        loss_D_real = loss.gan_loss(self.cfgs.gan_loss_mode, pred_real, True)
        ### fake
        pred_fake = netD(fake.detach())
        loss_D_fake = loss.gan_loss(self.cfgs.gan_loss_mode, pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.cfgs.lambda_identity
        lambda_A = self.cfgs.lambda_A
        lambda_B = self.cfgs.lambda_B

        if lambda_idt > 0:
            ### identity loss G_A: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_G_idt_A = loss.recon_loss(
                'l1', self.idt_A, self.real_B) * lambda_B * lambda_idt
            ### identity loss G_B: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_G_idt_B = loss.recon_loss(
                'l1', self.idt_B, self.real_A) * lambda_A * lambda_idt

        else:
            self.loss_G_idt_A = 0
            self.loss_G_idt_B = 0

        ### GAN loss D_A(G_A(A))
        self.loss_G_A = loss.gan_loss(self.cfgs.gan_loss_mode,
                                      self.netD_A(self.fake_B), True)
        ### GAN loss D_B(G_B(B))
        self.loss_G_B = loss.gan_loss(self.cfgs.gan_loss_mode,
                                      self.netD_B(self.fake_A), True)
        ### forward cycle loss ||G_B(G_A(A)) - A||
        self.loss_G_cycle_A = loss.recon_loss('l1', self.rec_A,
                                              self.real_A) * lambda_A
        ### backward cycle loss ||G_A(G_B(B)) - B||
        self.loss_G_cycle_B = loss.recon_loss('l1', self.rec_B,
                                              self.real_B) * lambda_B
        ### combine loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_cycle_A + self.loss_G_cycle_B + self.loss_G_idt_A + self.loss_G_idt_B

        self.loss_G.backward()

    def optimize_parameter(self):
        self.forward()
        self.set_stop_gradient([self.netD_A, self.netD_B], True)
        self.backward_G()  ## calculate gradients for G_A and G_B
        self.optimizer_G.optimizer.minimize(self.loss_G)
        self.optimizer_G.optimizer.clear_gradients()

        self.set_stop_gradient([self.netD_A], False)
        self.backward_D_A()  ### calculate gradients for D_A
        self.optimizer_D_A.optimizer.minimize(self.loss_D_A)
        self.optimizer_D_A.optimizer.clear_gradients()

        self.set_stop_gradient([self.netD_B], False)
        self.backward_D_B()  ### calculate gradients for D_B
        self.optimizer_D_B.optimizer.minimize(self.loss_D_B)
        self.optimizer_D_B.optimizer.clear_gradients()

    @paddle.no_grad()
    def test_single_side(self, direction):
        generator = getattr(self, 'netG_%s' % direction[0])
        self.fake_B = generator(self.real_A)

    def get_current_lr(self):
        lr_dict = {}
        lr_dict['optim_G'] = self.optimizer_G.optimizer.get_lr()
        lr_dict['optim_D_A'] = self.optimizer_D_A.optimizer.get_lr()
        lr_dict['optim_D_B'] = self.optimizer_D_B.optimizer.get_lr()
        return lr_dict

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.cfgs.save_dir, 'mobile', 'eval',
                                str(step))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.netG_A.eval()
        self.netG_B.eval()
        for direction in ['AtoB', 'BtoA']:
            eval_dataloader = getattr(self, 'eval_dataloader_' + direction)
            fakes = []
            cnt = 0
            for i, data_i in enumerate(eval_dataloader):
                self.set_single_input(data_i)
                self.test_single_side(direction)
                fakes.append(self.fake_B.detach().numpy())
                for j in range(len(self.fake_B)):
                    if cnt < 10:
                        name = 'fake_' + direction + str(i + j) + '.png'
                        save_path = os.path.join(save_dir, name)
                        fake_im = util.tensor2img(self.fake_B[j])
                        util.save_image(fake_im, save_path)
                    cnt += 1

            suffix = direction[-1]
            paddle.enable_static()
            fid = get_fid(fakes, self.inception_model,
                          getattr(self, 'npz_%s' % direction[-1]),
                          self.cfgs.inception_model)
            paddle.disable_static()
            if fid < getattr(self, 'best_fid_%s' % suffix):
                self.is_best = True
                setattr(self, 'best_fid_%s' % suffix, fid)
            print("direction: %s, fid score is: %f, best fid score is %f" %
                  (direction, fid, getattr(self, 'best_fid_%s' % suffix)))
            fids = getattr(self, 'fids_%s' % suffix)
            fids.append(fid)
            if len(fids) > 3:
                fids.pop(0)
            ret['metric/fid_%s' % suffix] = fid
            ret['metric/fid_%s-mean' %
                suffix] = sum(getattr(self, 'fids_%s' % suffix)) / len(
                    getattr(self, 'fids_%s' % suffix))
            ret['metric/fid_%s-best' % suffix] = getattr(self, 'best_fid_%s' %
                                                         suffix)

        self.netG_A.train()
        self.netG_B.train()
        return ret

import os
import numpy as np
import paddle.fluid as fluid
from distillers.base_resnet_distiller import BaseResnetDistiller
from models.super_modules import SuperConv2D
from models import loss
from configs.resnet_configs import get_configs
from metric import get_fid
from utils import util


class ResnetSupernet(BaseResnetDistiller):
    @staticmethod
    def add_special_cfgs(parser, load_pre=False):
        parser.add_argument(
            '--supernet_lr',
            type=float,
            default=2e-4,
            help="Initial learning rate to train super net")
        parser.add_argument(
            '--supernet_epoch',
            type=int,
            default=400,
            help="The number of epoch to train super net")
        parser.add_argument(
            '--supernet_nepochs',
            type=int,
            default=200,
            help="number of epochs with the initial learning rate")
        parser.add_argument(
            '--supernet_nepochs_decay',
            type=int,
            default=200,
            help="number of epochs to linearly decay learning rate to zero")
        parser.add_argument(
            '--supernet_scheduler',
            type=str,
            default='linear',
            help="learning rate scheduler in train supernet")
        parser.add_argument(
            '--supernet_student_netG',
            type=str,
            default='super_mobile_resnet_9blocks',
            help="Which student generator network to choose in supernet")
        parser.add_argument(
            '--config_set',
            type=str,
            default='channels-32',
            help="a set of configuration to get subnets of supernet")
        parser.add_argument(
            '--config_str',
            type=str,
            default=None,
            help="the configuration string used to get specific subnet of supernet"
        )
        if load_pre:
            super(ResnetSupernet, ResnetSupernet).add_special_cfgs(parser)
        return parser

    def __init__(self, cfgs):
        assert 'super' in cfgs.supernet_student_netG
        super(ResnetSupernet, self).__init__(cfgs, task='supernet')
        self.best_fid_largest = 1e9
        self.best_fid_smallest = 1e9
        self.fids_largest, self.fids_smallest = [], []
        if cfgs.config_set is not None:
            assert cfgs.config_str is None
            self.configs = get_configs(cfgs.config_set)
            self.cfgs.eval_mode = 'both'
        else:
            assert cfgs.config_str is not None
            self.configs = SingleConfigs(decode_config(cfgs.config_str))
            self.opt.eval_mode = 'largest'

    def forward(self, config):
        with fluid.dygraph.no_grad():
            self.Tfake_B = self.netG_teacher(self.real_A)
        self.Tfake_B.stop_gradient = True
        self.netG_student.configs = config
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2D)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            Sact = netA(Sact, {'channel': netA._num_filters})
            loss = fluid.layers.mse_loss(Sact, Tact)
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
        config = self.configs.sample()
        self.forward(config=config)
        self.set_stop_gradient(self.netD, False)
        self.backward_D()
        self.set_stop_gradient(self.netD, True)
        self.backward_G()

        self.optimizer_D.optimizer.minimize(self.loss_D)
        self.optimizer_D.optimizer.clear_gradients()
        self.optimizer_G.optimizer.minimize(self.loss_G)
        self.optimizer_G.optimizer.clear_gradients()

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.cfgs.save_dir, 'eval', str(step))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.netG_student.eval()
        if self.cfgs.eval_mode == 'both':
            setting = ('largest', 'smallest')
        else:
            setting = (self.cfgs.eval_mode, )
        for config_name in setting:
            config = self.configs(config_name)
            fakes, names = [], []
            for i, data_i in enumerate(self.eval_dataloader):
                id2name = self.name
                self.set_single_input(data_i)
                self.test(config)
                fakes.append(self.Sfake_B.detach().numpy())
                for j in range(len(self.Sfake_B)):
                    if i < 10:
                        Sname = 'Sfake_' + str(id2name[i + j]) + '.png'
                        Tname = 'Tfake_' + str(id2name[i + j]) + '.png'
                        Sfake_im = util.tensor2img(self.Sfake_B[j])
                        Tfake_im = util.tensor2img(self.Tfake_B[j])
                        util.save_image(Sfake_im,
                                        os.path.join(save_dir, Sname))
                        util.save_image(Tfake_im,
                                        os.path.join(save_dir, Tname))

            suffix = self.cfgs.direction
            fluid.disable_imperative()
            fid = get_fid(fakes, self.inception_model, self.npz,
                          self.cfgs.inception_model)
            fluid.enable_imperative()
            if fid < getattr(self, 'best_fid_%s' % config_name, fid):
                self.is_best = True
                setattr(self, 'best_fid_%s' % config_name, fid)
            fids = getattr(self, 'fids_%s' % config_name)
            fids.append(fid)
            if len(fids) > 3:
                fids.pop(0)
            ret['metric/fid_%s' % config_name] = fid
            ret['metric/fid_%s-mean' % config_name] = sum(
                getattr(self, 'fids_%s' % config_name)) / len(
                    getattr(self, 'fids_%s' % config_name))
            ret['metric/fid_%s-best' % config_name] = getattr(
                self, 'best_fid_%s' % config_name)
            print(
                "SuperNet Evalution config_name is : %s, fid score is: %f, best fid score is %f"
                %
                (config_name, fid, getattr(self, 'best_fid_%s' % config_name)))

        self.netG_student.train()
        return ret

    def test(self, config):
        with fluid.dygraph.no_grad():
            self.forward(config)

import os
import numpy as np
import itertools
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.base import to_variable
from models.super_modules import SuperConv2D
from models import loss
from models import network
from models.base_model import BaseModel
from utils import util, optimization
from dataset.data_loader import create_eval_data
from metric.inception import InceptionV3


class BaseResnetDistiller(BaseModel):
    @staticmethod
    def add_special_cfgs(parser, **kwcfgs):
        parser.add_argument(
            '--lambda_distill',
            type=float,
            default=0.01,
            help="the initialize lambda parameter for distiller loss")
        parser.add_argument(
            '--lambda_gan',
            type=float,
            default=1,
            help="the initialize lambda parameter for gan loss")
        parser.add_argument(
            '--lambda_recon',
            type=float,
            default=10.0,
            help="the initialize lambda parameter for recon loss")
        parser.add_argument(
            '--student_ngf',
            type=int,
            default=32,
            help="base channel of student generator")
        parser.add_argument(
            '--student_dropout_rate',
            type=float,
            default=0,
            help="dropout rate of student generator")
        parser.add_argument(
            '--restore_teacher_G_path',
            type=str,
            default=None,
            help="the pretrain model path of teacher generator")
        parser.add_argument(
            '--restore_student_G_path',
            type=str,
            default=None,
            help="the pretrain model path of student generator")
        parser.add_argument(
            '--real_stat_path',
            type=str,
            default='real_stat/horse2zebra_B.npz',
            help="path of real stat")
        return parser

    def __init__(self, cfgs, task):
        super(BaseResnetDistiller, self).__init__()
        self.cfgs = cfgs
        self.task = task
        self.loss_names = ['G_gan', 'G_distill', 'G_recon', 'D_fake', 'D_real']
        self.model_names = ['netG_student', 'netG_teacher', 'netD']
        self.netG_teacher = network.define_G(cfgs.input_nc, cfgs.output_nc,
                                             cfgs.ngf, cfgs.netG,
                                             cfgs.norm_type, cfgs.dropout_rate)
        student_netG = getattr(cfgs, self.task + '_student_netG')
        self.netG_student = network.define_G(
            cfgs.input_nc, cfgs.output_nc, cfgs.student_ngf, student_netG,
            cfgs.norm_type, cfgs.student_dropout_rate)
        if self.task == 'distiller':
            self.netG_pretrained = network.define_G(
                cfgs.input_nc, cfgs.output_nc, cfgs.pretrained_ngf,
                cfgs.pretrained_netG, cfgs.norm_type, 0)

        self.netD = network.define_D(cfgs.output_nc, cfgs.ndf, cfgs.netD,
                                     cfgs.norm_type, cfgs.n_layer_D)

        self.netG_teacher.eval()
        self.netG_student.train()
        self.netD.train()

        ### [9, 12, 15, 18]
        self.mapping_layers = ['model.%d' % i for i in range(9, 21, 3)]
        self.netAs = []
        self.Tacts, self.Sacts = {}, {}

        G_params = self.netG_student.parameters()
        for i, n in enumerate(self.mapping_layers):
            ft, fs = cfgs.ngf, cfgs.student_ngf
            if self.task == 'distiller':
                netA = Conv2D(
                    num_channels=fs * 4, num_filters=ft * 4, filter_size=1)
            else:
                netA = SuperConv2D(
                    num_channels=fs * 4, num_filters=ft * 4, filter_size=1)

            G_params += netA.parameters()
            self.netAs.append(netA)
            self.loss_names.append('G_distill%d' % i)

        if self.task == 'distiller':
            learning_rate = cfgs.distiller_lr
            scheduler = cfgs.distiller_scheduler
            nepochs = cfgs.distiller_nepochs
            nepochs_decay = cfgs.distiller_nepochs_decay
        elif self.task == 'supernet':
            learning_rate = cfgs.supernet_lr
            scheduler = cfgs.supernet_scheduler
            nepochs = cfgs.supernet_nepochs
            nepochs_decay = cfgs.supernet_nepochs_decay
        else:
            raise NotImplementedError("task {} is not suppport".format(
                self.task))

        self.optimizer_G = optimization.Optimizer(
            learning_rate,
            scheduler,
            cfgs.step_per_epoch,
            nepochs,
            nepochs_decay,
            cfgs,
            parameter_list=G_params)
        self.optimizer_D = optimization.Optimizer(
            learning_rate,
            scheduler,
            cfgs.step_per_epoch,
            nepochs,
            nepochs_decay,
            cfgs,
            parameter_list=self.netD.parameters())
        self.eval_dataloader, self.name = create_eval_data(
            cfgs, direction=cfgs.direction)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.is_best = False

        if cfgs.real_stat_path:
            self.npz = np.load(cfgs.real_stat_path)

        self.is_best = False

    def setup(self):
        self.load_networks()

        if self.cfgs.lambda_distill > 0:

            def get_activation(mem, name):
                def get_output_hook(layer, input, output):
                    mem[name] = output

                return get_output_hook

            def add_hook(net, mem, mapping_layers):
                for idx, (n, m) in enumerate(net.named_sublayers()):
                    if n in mapping_layers:
                        m.register_forward_post_hook(get_activation(mem, n))

        add_hook(self.netG_teacher, self.Tacts, self.mapping_layers)
        add_hook(self.netG_student, self.Sacts, self.mapping_layers)

    def set_input(self, inputs):
        self.real_A = inputs[0] if self.cfgs.direction == 'AtoB' else inputs[1]
        self.real_B = inputs[1] if self.cfgs.direction == 'AtoB' else inputs[0]

    def set_single_input(self, inputs):
        self.real_A = inputs[0]

    def load_networks(self):
        if self.cfgs.restore_teacher_G_path is None:
            if self.cfgs.direction == 'AtoB':
                teacher_G_path = os.path.join(self.cfgs.save_dir, 'mobile',
                                              'last_netG_A')
            else:
                teacher_G_path = os.path.join(self.cfgs.save_dir, 'mobile',
                                              'last_netG_B')
        else:
            teacher_G_path = self.cfgs.restore_teacher_G_path

        util.load_network(self.netG_teacher, teacher_G_path)

        if self.cfgs.restore_student_G_path is not None:
            util.load_network(self.netG_student,
                              self.cfgs.restore_student_G_path)
        else:
            if self.task == 'supernet':
                student_G_path = os.path.join(self.cfgs.save_dir, 'distiller',
                                              'last_stu_netG')
                util.load_network(self.netG_student, student_G_path)

        if self.cfgs.restore_D_path is not None:
            util.load_network(self.netD, self.cfgs.restore_D_path)
        if self.cfgs.restore_A_path is not None:
            for i, netA in enumerate(self.netAs):
                netA_path = '%s-%d.pth' % (self.cfgs.restore_A_path, i)
                util.load_network(netA, netA_path)
        if self.cfgs.restore_O_path is not None:
            util.load_optimizer(self.optimizer_G,
                                self.cfgs.restore_G_optimizer_path)
            util.load_optimizer(self.optimizer_D,
                                self.cfgs.restore_D_optimizer_path)

    def backward_D(self):
        fake = self.Sfake_B.detach()
        real = self.real_B.detach()

        pred_fake = self.netD(fake)
        self.loss_D_fake = loss.gan_loss(self.cfgs.gan_loss_mode, pred_fake,
                                         False)

        pred_real = self.netD(real)
        self.loss_D_real = loss.gan_loss(self.cfgs.gan_loss_mode, pred_real,
                                         True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def calc_distill_loss(self):
        raise NotImplementedError

    def backward_G(self):
        raise NotImplementedError

    def optimize_parameter(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def set_stop_gradient(self, nets, stop_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = stop_grad

    def save_network(self, epoch):
        save_filename = '{}_stu_netG'.format(epoch)
        save_path = os.path.join(self.cfgs.save_dir, self.task, save_filename)
        fluid.save_dygraph(self.netG_student.state_dict(), save_path)
        save_filename = '{}_tea_netG'.format(epoch)
        save_path = os.path.join(self.cfgs.save_dir, self.task, save_filename)
        fluid.save_dygraph(self.netG_teacher.state_dict(), save_path)

        save_filename = '{}_netD'.format(epoch)
        save_path = os.path.join(self.cfgs.save_dir, self.task, save_filename)
        fluid.save_dygraph(self.netD.state_dict(), save_path)

        for idx, net in enumerate(self.netAs):
            save_filename = '{}_netA-{}'.format(epoch, idx)
            save_path = os.path.join(self.cfgs.save_dir, self.task,
                                     save_filename)
            fluid.save_dygraph(net.state_dict(), save_path)

    def get_current_loss(self):
        loss_dict = {}
        for name in self.loss_names:
            if not hasattr(self, 'loss_' + name):
                continue
            key = name

            def has_num(key):
                for i in range(10):
                    if str(i) in key:
                        return True
                return False

            if has_num(key):
                continue
            loss_dict[key] = float(getattr(self, 'loss_' + name))
        return loss_dict

    def get_current_lr(self):
        lr_dict = {}
        lr_dict['optim_G'] = self.optimizer_G.optimizer.current_step_lr()
        lr_dict['optim_D'] = self.optimizer_D.optimizer.current_step_lr()
        return lr_dict

    def test(self):
        with fluid.dygraph.no_grad():
            self.forward()

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

import logging
import numpy as np
from collections import namedtuple
import paddle
import paddle.fluid as fluid
from .utils.utils import get_paddle_version, remove_model_fn
pd_ver = get_paddle_version()
if pd_ver == 185:
    from .layers_old import BaseBlock, SuperConv2D, SuperLinear
    Layer = paddle.fluid.dygraph.Layer
else:
    from .layers import BaseBlock, SuperConv2D, SuperLinear
    Layer = paddle.nn.Layer
from .utils.utils import search_idx
from ...common import get_logger
from ...core import GraphWrapper, dygraph2program
from .get_sub_model import get_prune_params_config, prune_params

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['OFA', 'RunConfig', 'DistillConfig']

RunConfig = namedtuple(
    'RunConfig',
    [
        # int, batch_size in training, used to get current epoch, default: None
        'train_batch_size',
        # list, the number of epoch of every task in training, default: None
        'n_epochs',
        # list, initial learning rate of every task in traning, NOT used now. Default: None.
        'init_learning_rate',
        # int, total images of train dataset, used to get current epoch, default: None
        'total_images',
        # list, elactic depth of the model in training, default: None
        'elastic_depth',
        # list, the number of sub-network to train per mini-batch data, used to get current epoch, default: None
        'dynamic_batch_size'
    ])
RunConfig.__new__.__defaults__ = (None, ) * len(RunConfig._fields)

DistillConfig = namedtuple(
    'DistillConfig',
    [
        # float, lambda scale of distillation loss, default: None.
        'lambda_distill',
        # instance of model, instance of teacher model, default: None.
        'teacher_model',
        # list(str), name of the layers which need a distillation, default: None.
        'mapping_layers',
        # str, the path of teacher pretrained model, default: None.
        'teacher_model_path',
        # instance of loss layer, the loss function used in distillation, if set to None, use mse_loss default, default: None.
        'distill_fn',
        # str, define which op append between teacher model and student model used in distillation, choice in ['conv', 'linear', None], default: None.
        'mapping_op'
    ])
DistillConfig.__new__.__defaults__ = (None, ) * len(DistillConfig._fields)


class OFABase(Layer):
    def __init__(self, model):
        super(OFABase, self).__init__()
        self.model = model
        self._layers, self._elastic_task = self.get_layers()

    def get_layers(self):
        layers = dict()
        elastic_task = set()
        for name, sublayer in self.model.named_sublayers():
            if isinstance(sublayer, BaseBlock):
                sublayer.set_supernet(self)
                if not sublayer.fixed:
                    layers[sublayer.key] = sublayer.candidate_config
                    for k in sublayer.candidate_config.keys():
                        elastic_task.add(k)
        return layers, elastic_task

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def layers_forward(self, block, *inputs, **kwargs):
        if getattr(self, 'current_config', None) != None:
            ### if block is fixed, donnot join key into candidate
            ### concrete config as parameter in kwargs
            if block.fixed == False:
                assert block.key in self.current_config, 'DONNT have {} layer in config.'.format(
                    block.key)
                config = self.current_config[block.key]
            else:
                config = dict()
                config.update(kwargs)
        else:
            config = dict()
        logging.debug(self.model, config)

        return block.fn(*inputs, **config)

    @property
    def layers(self):
        return self._layers


class OFA(OFABase):
    """
    Convert the training progress to the Once-For-All training progress, a detailed description in the paper: `Once-for-All: Train One Network and Specialize it for Efficient Deployment<https://arxiv.org/abs/1908.09791>`_ . This paper propose a training propgress named progressive shrinking (PS), which means we start with training the largest neural network with the maximum kernel size (i.e., 7), depth (i.e., 4), and width (i.e., 6). Next, we progressively fine-tune the network to support smaller sub-networks by gradually adding them into the sampling space (larger sub-networks may also be sampled). Specifically, after training the largest network, we first support elastic kernel size which can choose from {3, 5, 7} at each layer, while the depth and width remain the maximum values. Then, we support elastic depth and elastic width sequentially. 

    Parameters:
        model(paddle.nn.Layer): instance of model.
        run_config(paddleslim.ofa.RunConfig, optional): config in ofa training, can reference `<>`_ . Default: None.
        distill_config(paddleslim.ofa.DistillConfig, optional): config of distilltion in ofa training, can reference `<>`_. Default: None.
        elastic_order(list, optional): define the training order, if it set to None, use the default order in the paper. Default: None.
        train_full(bool, optional): whether to train the largest sub-network only. Default: False.

    Examples:
        .. code-block:: python
          from paddle.vision.models import mobilenet_v1
          from paddleslim.nas.ofa import OFA
          from paddleslim.nas.ofa.convert_super import Convert, supernet

          model = mobilenet_v1()
          sp_net_config = supernet(kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4])
          sp_model = Convert(sp_net_config).convert(model)
          ofa_model = OFA(sp_model)

    """

    def __init__(self,
                 model,
                 run_config=None,
                 distill_config=None,
                 elastic_order=None,
                 train_full=False):
        super(OFA, self).__init__(model)
        self.net_config = None
        self.run_config = run_config
        self.distill_config = distill_config
        self.elastic_order = elastic_order
        self.train_full = train_full
        self.iter = 0
        self.dynamic_iter = 0
        self.manual_set_task = False
        self.task_idx = 0
        self._add_teacher = False
        self.netAs_param = []

        ### if elastic_order is none, use default order
        if self.elastic_order is not None:
            assert isinstance(self.elastic_order,
                              list), 'elastic_order must be a list'

            if getattr(self.run_config, 'elastic_depth', None) != None:
                depth_list = list(set(self.run_config.elastic_depth))
                depth_list.sort()
                self.layers['depth'] = depth_list

        if self.elastic_order is None:
            self.elastic_order = []
            # zero, elastic resulotion, write in demo
            # first, elastic kernel size
            if 'kernel_size' in self._elastic_task:
                self.elastic_order.append('kernel_size')

            # second, elastic depth, such as: list(2, 3, 4)
            if getattr(self.run_config, 'elastic_depth', None) != None:
                depth_list = list(set(self.run_config.elastic_depth))
                depth_list.sort()
                self.layers['depth'] = depth_list
                self.elastic_order.append('depth')

            # final, elastic width
            if 'expand_ratio' in self._elastic_task:
                self.elastic_order.append('width')

            if 'channel' in self._elastic_task and 'width' not in self.elastic_order:
                self.elastic_order.append('width')

        if getattr(self.run_config, 'n_epochs', None) != None:
            assert len(self.run_config.n_epochs) == len(self.elastic_order)
            for idx in range(len(run_config.n_epochs)):
                assert isinstance(
                    run_config.n_epochs[idx],
                    list), "each candidate in n_epochs must be list"

            if self.run_config.dynamic_batch_size != None:
                assert len(self.run_config.n_epochs) == len(
                    self.run_config.dynamic_batch_size)
            if self.run_config.init_learning_rate != None:
                assert len(self.run_config.n_epochs) == len(
                    self.run_config.init_learning_rate)
                for idx in range(len(run_config.n_epochs)):
                    assert isinstance(
                        run_config.init_learning_rate[idx], list
                    ), "each candidate in init_learning_rate must be list"

        ### =================  add distill prepare ======================
        if self.distill_config != None:
            self._add_teacher = True
            self._prepare_distill()

        self.model.train()

    def _prepare_distill(self):
        if self.distill_config.teacher_model == None:
            logging.error(
                'If you want to add distill, please input instance of teacher model'
            )

        ### instance model by user can input super-param easily.
        assert isinstance(self.distill_config.teacher_model, Layer)

        # load teacher parameter
        if self.distill_config.teacher_model_path != None:
            param_state_dict, _ = paddle.load_dygraph(
                self.distill_config.teacher_model_path)
            self.distill_config.teacher_model.set_dict(param_state_dict)

        self.ofa_teacher_model = OFABase(self.distill_config.teacher_model)
        self.ofa_teacher_model.model.eval()

        # add hook if mapping layers is not None
        # if mapping layer is None, return the output of the teacher model,
        # if mapping layer is NOT None, add hook and compute distill loss about mapping layers.
        mapping_layers = getattr(self.distill_config, 'mapping_layers', None)
        if mapping_layers != None:
            self.netAs = []
            for name, sublayer in self.model.named_sublayers():
                if name in mapping_layers:
                    if self.distill_config.mapping_op != None:
                        if self.distill_config.mapping_op.lower() == 'conv2d':
                            netA = SuperConv2D(
                                getattr(sublayer, '_num_filters',
                                        sublayer._out_channels),
                                getattr(sublayer, '_num_filters',
                                        sublayer._out_channels), 1)
                        elif self.distill_config.mapping_op.lower() == 'linear':
                            netA = SuperLinear(
                                getattr(sublayer, '_output_dim',
                                        sublayer._out_features),
                                getattr(sublayer, '_output_dim',
                                        sublayer._out_features))
                        else:
                            raise NotImplementedError(
                                "Not Support Op: {}".format(
                                    self.distill_config.mapping_op.lower()))
                    else:
                        netA = None

                    if netA != None:
                        self.netAs_param.extend(netA.parameters())
                    self.netAs.append(netA)

    def _reset_hook_before_forward(self):
        self.Tacts, self.Sacts = {}, {}
        mapping_layers = getattr(self.distill_config, 'mapping_layers', None)
        if mapping_layers != None:

            def get_activation(mem, name):
                def get_output_hook(layer, input, output):
                    mem[name] = output

                return get_output_hook

            def add_hook(net, mem, mapping_layers):
                for idx, (n, m) in enumerate(net.named_sublayers()):
                    if n in mapping_layers:
                        m.register_forward_post_hook(get_activation(mem, n))

            add_hook(self.model, self.Sacts, mapping_layers)
            add_hook(self.ofa_teacher_model.model, self.Tacts, mapping_layers)

    def _compute_epochs(self):
        if getattr(self, 'epoch', None) == None:
            assert self.run_config.total_images is not None, \
                "if not use set_epoch() to set epoch, please set total_images in run_config."
            assert self.run_config.train_batch_size is not None, \
                "if not use set_epoch() to set epoch, please set train_batch_size in run_config."
            assert self.run_config.n_epochs is not None, \
                "if not use set_epoch() to set epoch, please set n_epochs in run_config."
            self.iter_per_epochs = self.run_config.total_images // self.run_config.train_batch_size
            epoch = self.iter // self.iter_per_epochs
        else:
            epoch = self.epoch
        return epoch

    def _sample_from_nestdict(self, cands, sample_type, task, phase):
        sample_cands = dict()
        for k, v in cands.items():
            if isinstance(v, dict):
                sample_cands[k] = self._sample_from_nestdict(
                    v, sample_type=sample_type, task=task, phase=phase)
            elif isinstance(v, list) or isinstance(v, set) or isinstance(v,
                                                                         tuple):
                if sample_type == 'largest':
                    sample_cands[k] = v[-1]
                elif sample_type == 'smallest':
                    sample_cands[k] = v[0]
                else:
                    if k not in task:
                        # sort and deduplication in candidate_config
                        # fixed candidate not in task_list
                        sample_cands[k] = v[-1]
                    else:
                        # phase == None -> all candidate; phase == number, append small candidate in each phase
                        # phase only affect last task in current task_list
                        if phase != None and k == task[-1]:
                            start = -(phase + 2)
                        else:
                            start = 0
                        sample_cands[k] = np.random.choice(v[start:])

        return sample_cands

    def _sample_config(self, task, sample_type='random', phase=None):
        config = self._sample_from_nestdict(
            self.layers, sample_type=sample_type, task=task, phase=phase)
        return config

    def set_task(self, task, phase=None):
        """
        set task in the ofa training progress.
        Parameters:
            task(list(str)|str): spectial task in training progress.
            phase(int, optional): the search space is gradually increased, use this parameter to spectial the phase in current task, if set to None, means use the whole search space in training progress. Default: None.
        Examples:
            .. code-block:: python
              ofa_model.set_task('width')
        """
        self.manual_set_task = True
        self.task = task
        self.phase = phase

    def set_epoch(self, epoch):
        """
        set epoch in the ofa training progress.
        Parameters:
            epoch(int): spectial epoch in training progress.
        Examples:
            .. code-block:: python
              ofa_model.set_epoch(3)
        """
        self.epoch = epoch

    def _progressive_shrinking(self):
        epoch = self._compute_epochs()
        self.task_idx, phase_idx = search_idx(epoch, self.run_config.n_epochs)
        self.task = self.elastic_order[:self.task_idx + 1]
        if 'width' in self.task:
            ### change width in task to concrete config
            self.task.remove('width')
            if 'expand_ratio' in self._elastic_task:
                self.task.append('expand_ratio')
            if 'channel' in self._elastic_task:
                self.task.append('channel')
        if len(self.run_config.n_epochs[self.task_idx]) == 1:
            phase_idx = None
        return self._sample_config(task=self.task, phase=phase_idx)

    def calc_distill_loss(self):
        """
        Calculate distill loss if there are distillation.
        Examples:
            .. code-block:: python
              dis_loss = ofa_model.calc_distill_loss()
        """
        losses = []
        assert len(self.netAs) > 0
        for i, netA in enumerate(self.netAs):
            n = self.distill_config.mapping_layers[i]
            ### add for elastic depth
            if n not in self.Sacts.keys():
                continue
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            if isinstance(netA, SuperConv2D):
                Sact = netA(
                    Sact,
                    channel=getattr(netA, '_num_filters', netA._out_channels))
            elif isinstance(netA, SuperLinear):
                Sact = netA(
                    Sact,
                    channel=getattr(netA, '_output_dim', netA._out_features))
            else:
                Sact = Sact

            Sact = Sact[0] if isinstance(Sact, tuple) else Sact
            Tact = Tact[0] if isinstance(Tact, tuple) else Tact
            if self.distill_config.distill_fn == None:
                loss = fluid.layers.mse_loss(Sact, Tact.detach())
            else:
                loss = distill_fn(Sact, Tact.detach())
            losses.append(loss)
        if self.distill_config.lambda_distill != None:
            return sum(losses) * self.distill_config.lambda_distill
        return sum(losses)

    ### TODO: complete it
    def search(self, eval_func, condition):
        pass

    def _export_sub_model_config(self, origin_model, config, input_shapes,
                                 input_dtypes):
        super_model_config = {}
        for name, sublayer in self.model.named_sublayers():
            if isinstance(sublayer, BaseBlock):
                for param in sublayer.parameters():
                    super_model_config[name] = sublayer.key

        for name, value in super_model_config.items():
            super_model_config[name] = config[value] if value in config.keys(
            ) else {}

        origin_model_config = {}
        for name, sublayer in origin_model.named_sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if name in super_model_config.keys():
                    origin_model_config[param.name] = super_model_config[name]

        program = dygraph2program(
            origin_model, inputs=input_shapes, dtypes=input_dtypes)
        graph = GraphWrapper(program)
        param_prune_config = get_prune_params_config(graph, origin_model_config)
        return param_prune_config

    def export(self,
               origin_model,
               config,
               input_shapes,
               input_dtypes,
               load_weights_from_supernet=True):
        """
        Export the weights according origin model and sub model config.
        Parameters:
            origin_model(paddle.nn.Layer): the instance of original model.
            config(dict): the config of sub model, can get by OFA.get_current_config() or some special config, such as paddleslim.nas.ofa.utils.dynabert_config(width_mult).
            input_shapes(list|list(list)): the shape of all inputs.
            input_dtypes(list): the dtype of all inputs.
            load_weights_from_supernet(bool, optional): whether to load weights from SuperNet. Default: False.
        Examples:
            .. code-block:: python
              from paddle.vision.models import mobilenet_v1
              origin_model = mobilenet_v1()

              config = {'conv2d_0': {'expand_ratio': 2}, 'conv2d_1': {'expand_ratio': 2}}
              origin_model = ofa_model.export(origin_model, config, input_shapes=[1, 3, 28, 28], input_dtypes=['float32'])
        """
        super_sd = None
        if load_weights_from_supernet:
            super_sd = remove_model_fn(origin_model, self.model.state_dict())

        param_config = self._export_sub_model_config(origin_model, config,
                                                     input_shapes, input_dtypes)
        prune_params(origin_model, param_config, super_sd)
        return origin_model

    @property
    def get_current_config(self):
        return self.current_config

    def set_net_config(self, net_config):
        """
        Set the config of the special sub-network to be trained.
        Parameters:
            net_config(dict): special the config of sug-network.
        Examples:
            .. code-block:: python
              config = {'conv2d_0': {'expand_ratio': 2}, 'conv2d_1': {'expand_ratio': 2}}
              ofa_model.set_net_config(config)
        """
        self.net_config = net_config

    def forward(self, *inputs, **kwargs):
        # =====================  teacher process  =====================
        teacher_output = None
        if self._add_teacher:
            self._reset_hook_before_forward()
            teacher_output = self.ofa_teacher_model.model.forward(*inputs,
                                                                  **kwargs)
        # ============================================================

        # ====================   student process  =====================
        if getattr(self.run_config, 'dynamic_batch_size', None) != None:
            self.dynamic_iter += 1
            if self.dynamic_iter == self.run_config.dynamic_batch_size[
                    self.task_idx]:
                self.iter += 1
                self.dynamic_iter = 0

        if self.net_config == None:
            if self.train_full == True:
                self.current_config = self._sample_config(
                    task=None, sample_type='largest')
            else:
                if self.manual_set_task == False:
                    self.current_config = self._progressive_shrinking()
                else:
                    self.current_config = self._sample_config(
                        self.task, phase=self.phase)
        else:
            self.current_config = self.net_config

        _logger.debug("Current config is {}".format(self.current_config))
        if 'depth' in self.current_config:
            kwargs['depth'] = self.current_config['depth']

        return self.model.forward(*inputs, **kwargs), teacher_output

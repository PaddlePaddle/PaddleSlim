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

import numpy as np
from collections import namedtuple
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from layers import BaseBlock, Block, SuperConv2D, SuperBatchNorm
from utils.utils import search_idx

__all__ = ['OFA', 'RunConfig']

RunConfig = namedtuple('RunConfig', [
    'train_batch_size', 'eval_batch_size', 'n_epochs', 'save_frequency',
    'eval_frequency', 'init_learning_rate', 'total_images', 'elastic_depth',
    'dynamic_batch_size'
])
RunConfig.__new__.__defaults__ = (None, ) * len(RunConfig._fields)


class OFABase(fluid.dygraph.Layer):
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
                layers[sublayer.key] = sublayer.candidate_config
                for k in sublayer.candidate_config.keys():
                    elastic_task.add(k)
        return layers, elastic_task

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def layers_forward(self, block, *inputs, **kwargs):
        assert block.key in self.current_config, 'DONNT have {} layer in config.'.format(
            block.key)
        config = self.current_config[block.key]
        return block.fn(*inputs, **config)

    @property
    def layers(self):
        return self._layers


class OFA(OFABase):
    def __init__(self,
                 model,
                 run_config,
                 config=None,
                 lambda_distill=0,
                 elastic_order=None,
                 train_full=False):
        super(OFA, self).__init__(model)
        self.run_config = run_config
        self.config = config
        self.elastic_order = elastic_order
        self.lambda_distill = lambda_distill
        self.train_full = train_full
        self.iter_per_epochs = 1  #self.run_config.total_images // self.run_config.train_batch_size
        self.iter = 0
        self.dynamic_iter = 0
        self.manual_set_task = False
        self.task_idx = 0

        for idx in range(len(run_config.n_epochs)):
            assert isinstance(
                run_config.init_learning_rate[idx],
                list), "each candidate in init_learning_rate must be list"
            assert isinstance(run_config.n_epochs[idx],
                              list), "each candidate in n_epochs must be list"

        ### if elastic_order is none, use default order
        if self.elastic_order is not None:
            assert isinstance(self.elastic_order,
                              list), 'elastic_order must be a list'

        if self.elastic_order is None:
            self.elastic_order = []
            # zero, elastic resulotion, write in demo
            # first, elastic kernel size
            if 'kernel_size' in self._elastic_task:
                self.elastic_order.append('kernel_size')

            # second, elastic depth, depth -> list(2, 3, 4)
            if getattr(self.run_config, 'elastic_depth', None) != None:
                self.elastic_order.append('depth')

            # final, elastic width
            if 'expand_ratio' in self._elastic_task:
                self.elastic_order.append('expand_ratio')

            if 'channel' in self._elastic_task:
                self.elastic_order.append('channel')

    ### TODO: complete it
    #    self.model = model()
    #    if lambda_distill > 0:
    #        self.teacher_model = model(config=max_config)
    #        self.teacher_model.eval()
    #    self.model.train()
    #    self.elastic_order = elastic_order

    #if self.elastic_order == None:
    #self.elastic_order = self.model.__dict__

    def _compute_epochs(self):
        if getattr(self, 'epoch', None) == None:
            epoch = self.iter // self.iter_per_epochs
        else:
            epoch = self.epochs
        return epoch

    def _sample_from_nestdict(self, cands, sample_type, task, phase):
        sample_cands = dict()
        for k, v in cands.items():
            if isinstance(v, dict):
                sample_cands[k] = self._sample_from_nestdict(
                    v, sample_type=sample_type, task=task, phase=phase)
            elif isinstance(v, list) or isinstance(v, set):
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

    ### TODO: if task is elastic width, need to add re_organize_middle_weight in 1x1 conv
    def set_task(self, task=None, phase=None):
        self.manual_set_task = True
        self.task = task
        self.phase = phase

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _progressive_shrinking(self):
        epoch = self._compute_epochs()
        self.task_idx, phase_idx = search_idx(epoch, self.run_config.n_epochs)
        self.task = self.elastic_order[:self.task_idx + 1]
        if len(self.run_config.n_epochs[self.task_idx]) == 1:
            phase_idx = None
        return self._sample_config(task=self.task, phase=phase_idx)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2D)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            Sact = netA(Sact)
            #loss = fluid.layers.mse_loss(Sact, Tact)
            loss = distill_fn(Sact, Tact)
            losses.append(loss)
        return sum(losses)

    ### TODO: complete it
    def search(self, eval_func, condition):
        pass

    ### TODO: complete it
    def export(self, config):
        pass

    def forward(self, *inputs, **kwargs):
        self.dynamic_iter += 1
        if self.dynamic_iter == self.run_config.dynamic_batch_size[
                self.task_idx]:
            self.iter += 1
            self.dynamic_iter = 0

        if self.config == None:
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
            self.current_config = self.config
        return self.model.forward(*inputs, **kwargs)


class SuperNet(fluid.dygraph.Layer):
    def __init__(self):
        super(SuperNet, self).__init__()
        models = [
            Block(
                SuperConv2D(
                    3,
                    5,
                    7,
                    candidate_config={
                        'kernel_size': [7, 5, 7, 3],
                        'expand_ratio': [2, 3, 4]
                    },
                    transform_kernel=True),
                key='conv1')
        ]
        models += [SuperBatchNorm(20)]
        models += [nn.Pool2D(global_pooling=True)]
        models += [nn.ReLU()]
        models += [
            Block(
                SuperConv2D(
                    5,
                    5,
                    7,
                    candidate_config={
                        'kernel_size': [3, 5, 7],
                        'expand_ratio': [2, 3, 4]
                    },
                    transform_kernel=True),
                key='conv2')
        ]
        models += [nn.Pool2D(global_pooling=True)]
        models += [nn.ReLU()]
        self.models = nn.Sequential(*models)

    def forward(self, x):
        return self.models(x)


# NOTE: case 1
if __name__ == '__main__':
    data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
    label_np = np.random.random((1)).astype(np.float32)

    default_run_config = {
        'train_batch_size': 256,
        'eval_batch_size': 64,
        'n_epochs': [[5], [20, 40]],
        'init_learning_rate': [[0.001], [0.0001, 0.003]],
        'dynamic_batch_size': [1, 1],
        'total_images': 1281167
    }
    run_config = RunConfig(**default_run_config)

    assert len(run_config.n_epochs) == len(run_config.dynamic_batch_size)
    assert len(run_config.n_epochs) == len(run_config.init_learning_rate)

    fluid.enable_dygraph()
    model = SuperNet()
    ofa_model = OFA(model, run_config)
    print(ofa_model.state_dict().keys())

    data = fluid.dygraph.to_variable(data_np)
    label = fluid.dygraph.to_variable(label_np)

    start_epoch = 0
    for idx in range(len(run_config.n_epochs)):
        cur_idx = run_config.n_epochs[idx]
        for ph_idx in range(len(cur_idx)):
            cur_lr = run_config.init_learning_rate[idx][ph_idx]
            adam = fluid.optimizer.Adam(
                learning_rate=cur_lr, parameter_list=ofa_model.parameters())
            for epoch_id in range(start_epoch,
                                  run_config.n_epochs[idx][ph_idx]):
                # add for data in dataset:
                for model_no in range(run_config.dynamic_batch_size[idx]):
                    output = ofa_model(data)
                    loss = fluid.layers.reduce_mean(output)
                    print('epoch: {}, loss: {}'.format(epoch_id,
                                                       loss.numpy()[0]))
                    loss.backward()
                    adam.minimize(loss)
                    adam.clear_gradients()
            start_epoch = run_config.n_epochs[idx][ph_idx]

# NOTE: case 2
#class BaseManager:
#    def __init__(self, model, run_config, optim_fn, train_dataset, eval_dataset):
#        self.model = model
#        self.run_config = run_config
#        self.loss = loss_fn
#        ### optim_fn = dict('Adam': parameter_list: PARAM, learning_rate) ??? 
#        self.optim = optim_fn
#        self.train_dataset = train_dataset
#        self.eval_dataset = eval_dataset
#
#    def train_one_epoch(self):
#        for image, label in self.train_dataset():
#            out = self.model(image)
#            loss = self.loss(out, label)
#            self.optim.clear_gradient()
#            loss.backward()
#            self.optim.minimize(loss)
#
#    def eval_one_epoch(self):
#        for image, label in self.eval_dataset():
#            out = self.model(image)
#            acc_top1, acc_top5 = accuracy(out, label)
#
#        # compute final acc
#
#if name == '__main__':
#    data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
#    label_np = np.random.random((1)).astype(np.float32)
#
#    fluid.enable_dygraph()
#    model = SuperNet()
#    run_config = dict(#TODO)
#    my_manager = BaseManager(model, run_config)
#    ofa_model = OFA(my_manager)
#    ofa_mode.train()

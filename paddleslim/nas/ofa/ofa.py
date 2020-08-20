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
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from paddleslim.core.layers import BaseBlock, Block, SuperConv2D

__all__ = ['OFA']


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

    def layers_forward(self, blocks, *inputs, **kwargs):
        raise NotImplementedError

    @property
    def layers(self):
        return self._layers


class OFA(OFABase):
    def __init__(self,
                 model,
                 config=None,
                 lambda_distill=0,
                 elastic_order=None,
                 train_full=False):
        super(OFA, self).__init__(model)
        self.config = config
        self.elastic_order = elastic_order
        self.lambda_distill = lambda_distill
        self.train_full = train_full

        ### if elastic_order is none, use default order
        ### TODO: consider channel and expand_stdio
        if self.elastic_order is not None:
            assert isinstance(self.elastic_order,
                              list), 'elastic_order must be a list'

        if self.elastic_order is None:
            self.elastic_order = []
            if 'kernel_size' in self._elastic_task:
                self.elastic_order.append('kernel_size')
            if 'expand_stdio' in self._elastic_task:
                self.elastic_order.append('width')

        ### if not set_epoch, start it from 0
        self.epoch = 0
        self.task_num = len(self.elastic_order)

    ### TODO: complete it
    #    self.model = model()
    #    if lambda_distill > 0:
    #        self.teacher_model = model(config=max_config)
    #        self.teacher_model.eval()
    #    self.model.train()
    #    self.elastic_order = elastic_order

    #if self.elastic_order == None:
    #self.elastic_order = self.model.__dict__

    def _sample_from_nestdict(self, cands, sample_type):
        sample_cands = dict()
        for k, v in cands.items():
            if isinstance(v, dict):
                sample_cands[k] = self._sample_from_nestdict(
                    v, sample_type=sample_type)
            elif isinstance(v, list) or isinstance(v, set):
                if sample_type == 'largest':
                    sample_cand = v[-1]
                elif sample_type == 'smallest':
                    sample_cand = v[0]
                else:
                    if self.task == 'kernel_size' and k != self.task:
                        ### sort and deduplication in candidate_config
                        sample_cands[k] = v[-1]
                    else:
                        if self.phase == 0:
                            sample_cands[k] = np.random.choice(v[1:])
                        else:
                            sample_cands[k] = np.random.choice(v)

        return sample_cands

    def _sample_config(self, sample_type='random'):
        config = self._sample_from_nestdict(
            self.layers, sample_type=sample_type)
        return config

    def layers_forward(self, block, *inputs, **kwargs):
        assert block.key in self.current_config, 'DONNT have {} layer in config.'.format(
            block.key)
        config = self.current_config[block.key]
        return block.fn(*inputs, **config)

    ### if task is elastic width, need to add re_organize_middle_weight in 1x1 conv
    def set_task(self, task=None):
        self.task = task

    def set_epoch(self, epoch):
        self.epoch = epoch

    ### TODO: complete it
    def _progressive_shrinking(self):
        ### judge the length of elastic_order
        self.phase = 0
        if self.epoch < 0:
            self.task = self.elastic_order[0]
        else:
            self.task = self.elastic_order[1]
            if self.epoch < 15:
                self.phase = 0
            else:
                self.phase = 1

        return self._sample_config()

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2D)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            Sact = netA(Sact)
            loss = fluid.layers.mse_loss(Sact, Tact)
            losses.append(loss)
        return sum(losses)

    ### TODO: complete it
    def search(self, eval_func, condition):
        pass

    ### TODO: complete it
    def export(self, config):
        pass

    def forward(self, *inputs, **kwargs):
        self.epoch += 1
        if self.config == None:
            if self.train_full == True:
                self.current_config = self._sample_config(
                    self.layers, sample_type='largest')
            else:
                self.current_config = self._progressive_shrinking()
        else:
            self.current_config = self.config
        #print(self.current_config, self.layers)
        return self.model.forward(*inputs, **kwargs)


class SuperNet(fluid.dygraph.Layer):
    def __init__(self):
        super(SuperNet, self).__init__()
        models = [#Conv2D(3, 5, 7)]
            Block(
                SuperConv2D(
                    3,
                    5,
                    7,
                    candidate_config={
                        'kernel_size': [7, 5, 7, 3],
                        'expand_stdio': [2, 3, 4]
                    },
                    transform_kernel='1'),
                key='conv1')
        ]
        #        models += [nn.BatchNorm(5)]
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
                        'expand_stdio': [2, 3, 4]
                    },
                    transform_kernel='1'),
                key='conv2')
        ]
        models += [nn.Pool2D(global_pooling=True)]
        models += [nn.ReLU()]
        self.models = nn.Sequential(*models)

    def forward(self, x):
        return self.models(x)


if __name__ == '__main__':
    data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
    label_np = np.random.random((1)).astype(np.float32)

    fluid.enable_dygraph()
    model = SuperNet()
    ofa_model = OFA(model)
    print(ofa_model.state_dict().keys())

    data = fluid.dygraph.to_variable(data_np)
    label = fluid.dygraph.to_variable(label_np)
    adam = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=ofa_model.parameters())

    for _ in range(2):
        output = ofa_model(data)
        loss = fluid.layers.reduce_mean(output)
        print('loss: {}'.format(loss.numpy()[0]))
        loss.backward()
        adam.minimize(loss)
        adam.clear_gradients()

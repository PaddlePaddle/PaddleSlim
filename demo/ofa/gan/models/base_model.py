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
import paddle.nn as nn


class BaseModel(nn.Layer):
    @staticmethod
    def add_special_cfgs(parser):
        raise NotImplementedError

    def set_input(self, inputs):
        raise NotImplementedError

    def setup(self, model_weight=None):
        self.load_network()

    def load_network(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.args, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path)

    def save_network(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s' % (epoch, name)
                save_path = os.path.join(self.args.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                paddle.save(net.state_dict(), save_path)

    def forward(self):
        raise NotImplementedError

    def optimize_parameter(self):
        raise NotImplementedError

    def get_current_loss(self):
        loss_dict = {}
        for name in self.loss_names:
            if not hasattr(self, 'loss_' + name):
                continue
            key = name
            loss_dict[key] = float(getattr(self, 'loss_' + name))
        return loss_dict

    def get_current_lr(self):
        raise NotImplementedError

    def set_stop_gradient(self, nets, stop_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = stop_grad

    def evaluate_model(self):
        raise NotImplementedError

    def profile(self):
        raise NotImplementedError

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
from functools import reduce
import paddle.fluid as fluid
import network
from utils import util
from paddleslim.analysis.flops import dygraph_flops


class TestModel(fluid.dygraph.Layer):
    def __init__(self, cfgs):
        super(TestModel, self).__init__()
        self.model_names = ['G']
        self.netG = network.define_G(cfgs.input_nc, cfgs.output_nc, cfgs.ngf,
                                     cfgs.netG, cfgs.norm_type,
                                     cfgs.dropout_rate)
        self.netG.eval()
        self.cfgs = cfgs

    def set_input(self, input):
        self.real_A = input[0]

    def setup(self):
        self.load_network()

    def load_network(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.cfgs, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path)

    def forward(self, config=None):
        if config is not None:
            self.netG.configs = config
        self.fake_B = self.netG(self.real_A)

    def test(self, config=None):
        with fluid.dygraph.no_grad():
            self.forward(config)

    def profile(self, config=None):
        netG = self.netG
        netG.configs = config
        with fluid.dygraph.no_grad():
            flops = dygraph_flops(
                netG, (self.real_A[:1]), only_conv=False, only_multiply=True)
        params = 0
        for p in netG.parameters():
            if 'instance_norm' in p.name:
                continue
            params += reduce(lambda x, y: x * y, p.shape)
        return flops, params

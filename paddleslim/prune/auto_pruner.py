# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import paddle.fluid as fluid
from .pruner import Pruner
from ..core import VarWrapper, OpWrapper, GraphWrapper

from ..search import ControllerServer

__all__ = ["AutoPruner"]


class AutoPruner(object):
    def __init__(self,
                 params=[],
                 init_ratios=None,
                 pruned_flops=0.5,
                 pruned_latency=None,
                 server_addr=("", ""),
                 search_strategy="sa"):
        """
        Search a group of ratios used to prune program.
        Args:
            params(list<str>): The names of parameters to be pruned.
            init_ratios(list<float>|float): Init ratios used to pruned parameters in `params`.
                                            List means ratios used for pruning each parameter in `params`.
                                            The length of `init_ratios` should be equal to length of params when `init_ratios` is a list. 
                                            If it is a scalar, all the parameters in `params` will be pruned by uniform ratio.
                                            None means get a group of init ratios by `pruned_flops` of `pruned_latency`. Default: None.
            pruned_flops(float): The percent of FLOPS to be pruned. Default: None.
            pruned_latency(float): The percent of latency to be pruned. Default: None.
            server_addr(tuple): A tuple of server ip and server port for controller server. 
            search_strategy(str): The search strategy. Default: 'sa'.
        """
        # step1: Create controller server. And start server if current host match server_ip.
        self._controller_server = ControllerServer(
            addr=(server_ip, server_port), search_strategy="sa")
        self._params = params
        self._init_ratios = init_ratios
        self._pruned_flops = pruned_flops
        self._pruned_latency = pruned_latency
        self._pruner = Pruner()
        self._controller_agent = None
        self._base_flops = None
        self._base_latency = None

    def prune(self, program, scope, place):

        if self._controller_agent is None:
            self._controller_agent = PrunerAgent(
                addr=self._controller_server.addr, self._range_table)
            if self._init_ratios is None:
                self._init_ratios = self._get_init_ratios(
                    program, self._params, self._pruned_flops,
                    self._pruned_latency)
            self._current_ratios = self._init_ratios
        else:
            self._current_ratios = self._controller_agent.next_ratios()

        if self._base_flops == None:
            self._base_flops = flops(program)

        for i in range(self._max_try_num):
            pruned_program = self._pruner.prune(
                program,
                scope,
                self._params,
                self._current_ratios,
                only_graph=True)
            if flops(pruned_program) < self._base_flops * (
                    1 - self._pruned_flops):
                break
            self._current_ratios = self._controller_agent.illegal_ratios(
                self._current_ratios)

        pruned_program = self._pruner.prune(program, scope, self._params,
                                            self._current_ratios)
        return pruned_program

    def reward(self, score):
        self._controller_agent.reward(self._current_ratios, score)


class PrunerAgent(object):
    """
    The agent used to talk with controller server.
    """

    def __init__(self, server_attr=("", ""), range_table):
        self._range_table = range_table
        self._controller_client = ControllerClient(server_attr)
        self._controller_client.send_range_table(range_table)

    def next_ratios(self):
        tokens = self._controller_client.next_tokens()
        self._tokens2ratios(tokens)

    def illegal_ratios(self, ratios):
        tokens = self._ratios2tokens(ratios)
        tokens = self._controller_client.illegal_tokens(tokens)
        return self._tokens2ratios(tokens)

    def _ratios2tokens(self, ratios):
        """Convert pruned ratios to tokens.
        """
        return [int(ratio / 0.01) for ratio in ratios]

    def _tokens2_ratios(self, tokens):
        """Convert tokens to pruned ratios.
        """
        return [token * 0.01 for token in tokens]

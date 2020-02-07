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

import socket
import logging
import numpy as np
import paddle.fluid as fluid
from .pruner import Pruner
from ..core import VarWrapper, OpWrapper, GraphWrapper
from ..common import SAController
from ..common import get_logger
from ..analysis import flops

from ..common import ControllerServer
from ..common import ControllerClient

__all__ = ["AutoPruner"]

_logger = get_logger(__name__, level=logging.INFO)


class AutoPruner(object):
    def __init__(self,
                 scope,
                 place,
                 params=None,
                 init_ratios=None,
                 server_addr=("", 0),
                 init_temperature=100,
                 reduce_rate=0.85,
                 max_try_times=300,
                 max_client_num=10,
                 search_steps=300,
                 max_ratios=[0.9],
                 min_ratios=[0],
                 key="auto_pruner",
                 is_server=True):
        """
        Search a group of ratios used to prune program.
        Args:
            scope(Scope): The scope to be pruned.
            place(fluid.Place): The device place of parameters.
            params(list<str>): The names of parameters to be pruned.
            init_ratios(list<float>|float): Init ratios used to pruned parameters in `params`.
                List means ratios used for pruning each parameter in `params`.
                The length of `init_ratios` should be equal to length of params when `init_ratios` is a list. 
                If it is a scalar, all the parameters in `params` will be pruned by uniform ratio.
                None means get a group of init ratios by `pruned_flops` of `pruned_latency`. Default: None.
            server_addr(tuple): A tuple of server ip and server port for controller server. 
            init_temperature(float): The init temperature used in simulated annealing search strategy.
            reduce_rate(float): The decay rate used in simulated annealing search strategy.
            max_try_times(int): The max number of trying to generate legal tokens.
            max_client_num(int): The max number of connections of controller server.
            search_steps(int): The steps of searching.
            max_ratios(float|list<float>): Max ratios used to pruned parameters in `params`.
                List means max ratios for each parameter in `params`.
                The length of `max_ratios` should be equal to length of params when `max_ratios` is a list.
                If it is a scalar, it will used for all the parameters in `params`.
            min_ratios(float|list<float>): Min ratios used to pruned parameters in `params`.
                List means min ratios for each parameter in `params`.
                The length of `min_ratios` should be equal to length of params when `min_ratios` is a list.
                If it is a scalar, it will used for all the parameters in `params`.
            key(str): Identity used in communication between controller server and clients.
            is_server(bool): Whether current host is controller server. Default: True.
        """

        self._scope = scope
        self._place = place
        self._params = params
        self._init_ratios = init_ratios
        assert (params is not None and init_ratios is not None)
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._max_try_times = max_try_times
        self._is_server = is_server

        self._range_table = self._get_range_table(min_ratios, max_ratios)

        self._pruner = Pruner()
        init_tokens = self._ratios2tokens(self._init_ratios)
        _logger.info("range table: {}".format(self._range_table))
        controller = SAController(self._range_table, self._reduce_rate,
                                  self._init_temperature, self._max_try_times,
                                  init_tokens)

        server_ip, server_port = server_addr
        if server_ip == None or server_ip == "":
            server_ip = self._get_host_ip()

        self._controller_server = ControllerServer(
            controller=controller,
            address=(server_ip, server_port),
            max_client_num=max_client_num,
            search_steps=search_steps,
            key=key)

        # create controller server
        if self._is_server:
            self._controller_server.start()

        self._controller_client = ControllerClient(
            self._controller_server.ip(),
            self._controller_server.port(),
            key=key)

        self._iter = 0
        self._param_backup = {}

    def _get_host_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def _get_range_table(self, min_ratios, max_ratios):
        assert isinstance(min_ratios, list) or isinstance(min_ratios, float)
        assert isinstance(max_ratios, list) or isinstance(max_ratios, float)
        min_ratios = min_ratios if isinstance(
            min_ratios, list) else [min_ratios] * len(self._params)
        max_ratios = max_ratios if isinstance(
            max_ratios, list) else [max_ratios] * len(self._params)
        min_tokens = self._ratios2tokens(min_ratios)
        max_tokens = self._ratios2tokens(max_ratios)
        return (min_tokens, max_tokens)

    def prune(self, program, eval_program=None):
        """
        Prune program with latest tokens generated by controller.
        Args:
            program(fluid.Program): The program to be pruned.
        Returns:
            Program: The pruned program.
        """
        self._current_ratios = self._next_ratios()
        pruned_program, _, _ = self._pruner.prune(
            program,
            self._scope,
            self._params,
            self._current_ratios,
            place=self._place,
            only_graph=False,
            param_backup=self._param_backup)
        pruned_val_program = None
        if eval_program is not None:
            pruned_val_program, _, _ = self._pruner.prune(
                program,
                self._scope,
                self._params,
                self._current_ratios,
                place=self._place,
                only_graph=True)

        _logger.info("AutoPruner - pruned ratios: {}".format(
            self._current_ratios))
        return pruned_program, pruned_val_program

    def reward(self, score):
        """
        Return reward of current pruned program.
        Args:
            score(float): The score of pruned program.
        """
        self._restore(self._scope)
        self._param_backup = {}
        tokens = self._ratios2tokens(self._current_ratios)
        self._controller_client.update(tokens, score, self._iter)
        self._iter += 1

    def _restore(self, scope):
        for param_name in self._param_backup.keys():
            param_t = scope.find_var(param_name).get_tensor()
            param_t.set(self._param_backup[param_name], self._place)

    def _next_ratios(self):
        tokens = self._controller_client.next_tokens()
        return self._tokens2ratios(tokens)

    def _ratios2tokens(self, ratios):
        """Convert pruned ratios to tokens.
        """
        return [int(ratio / 0.01) for ratio in ratios]

    def _tokens2ratios(self, tokens):
        """Convert tokens to pruned ratios.
        """
        return [token * 0.01 for token in tokens]

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
from ..core import VarWrapper, OpWrapper, GraphWrapper
from ..common import SAController
from ..common import get_logger
from ..analysis import flops

from ..common import ControllerServer
from ..common import ControllerClient
from .search_space import SearchSpaceFactory

__all__ = ["SANAS"]

_logger = get_logger(__name__, level=logging.INFO)


class SANAS(object):
    def __init__(self,
                 configs,
                 max_flops=None,
                 max_latency=None,
                 server_addr=("", 0),
                 init_temperature=100,
                 reduce_rate=0.85,
                 max_try_number=300,
                 max_client_num=10,
                 search_steps=300,
                 key="sa_nas",
                 is_server=True):
        """
        Search a group of ratios used to prune program.
        Args:
            configs(list<tuple>): A list of search space configuration with format (key, input_size, output_size, block_num).
                                  `key` is the name of search space with data type str. `input_size` and `output_size`  are
                                   input size and output size of searched sub-network. `block_num` is the number of blocks in searched network.
            max_flops(int): The max flops of searched network. None means no constrains. Default: None.
            max_latency(float): The max latency of searched network. None means no constrains. Default: None.
            server_addr(tuple): A tuple of server ip and server port for controller server. 
            init_temperature(float): The init temperature used in simulated annealing search strategy.
            reduce_rate(float): The decay rate used in simulated annealing search strategy.
            max_try_number(int): The max number of trying to generate legal tokens.
            max_client_num(int): The max number of connections of controller server.
            search_steps(int): The steps of searching.
            key(str): Identity used in communication between controller server and clients.
            is_server(bool): Whether current host is controller server. Default: True.
        """

        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._max_try_number = max_try_number
        self._is_server = is_server
        self._max_flops = max_flops
        self._max_latency = max_latency

        self._configs = configs

        factory = SearchSpaceFactory()
        self._search_space = factory.get_search_space(configs)
        init_tokens = self._search_space.init_tokens()
        range_table = self._search_space.range_table()
        range_table = (len(range_table) * [0], range_table)

        print range_table

        controller = SAController(range_table, self._reduce_rate,
                                  self._init_temperature, self._max_try_number,
                                  init_tokens, self._constrain_func)

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

    def _get_host_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def _constrain_func(self, tokens):
        if (self._max_flops is None) and (self._max_latency is None):
            return True
        archs = self._search_space.token2arch(tokens)
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            i = 0
            for config, arch in zip(self._configs, archs):
                input_size = config[1]["input_size"]
                input = fluid.data(
                    name="data_{}".format(i),
                    shape=[None, 3, input_size, input_size],
                    dtype="float32")
                output = arch(input)
                i += 1
        return flops(main_program) < self._max_flops

    def next_archs(self):
        """
        Get next network architectures.
        Returns:
            list<function>: A list of functions that define networks.
        """
        self._current_tokens = self._controller_client.next_tokens()
        archs = self._search_space.token2arch(self._current_tokens)
        return archs

    def reward(self, score):
        """
        Return reward of current searched network.
        Args:
            score(float): The score of current searched network.
        """
        self._controller_client.update(self._current_tokens, score)
        self._iter += 1

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

import os
import socket
import logging
import numpy as np
import json
import hashlib
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
                 server_addr=("", 8881),
                 init_temperature=100,
                 reduce_rate=0.85,
                 search_steps=300,
                 save_checkpoint='nas_checkpoint',
                 load_checkpoint=None,
                 is_server=False):
        """
        Search a group of ratios used to prune program.
        Args:
            configs(list<tuple>): A list of search space configuration with format [(key, {input_size, output_size, block_num, block_mask})].
                                  `key` is the name of search space with data type str. `input_size` and `output_size`  are
                                   input size and output size of searched sub-network. `block_num` is the number of blocks in searched network, `block_mask` is a list consists by 0 and 1, 0 means normal block, 1 means reduction block.
            server_addr(tuple): A tuple of server ip and server port for controller server. 
            init_temperature(float): The init temperature used in simulated annealing search strategy.
            reduce_rate(float): The decay rate used in simulated annealing search strategy.
            search_steps(int): The steps of searching.
            save_checkpoint(string|None): The directory of checkpoint to save, if set to None, not save checkpoint. Default: 'nas_checkpoint'.
            load_checkpoint(string|None): The directory of checkpoint to load, if set to None, not load checkpoint. Default: None.
            is_server(bool): Whether current host is controller server. Default: True.
        """
        if not is_server:
            assert server_addr[
                0] != "", "You should set the IP and port of server when is_server is False."
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._is_server = is_server
        self._configs = configs
        self._key = hashlib.md5(str(self._configs).encode("utf-8")).hexdigest()

        server_ip, server_port = server_addr
        if server_ip == None or server_ip == "":
            server_ip = self._get_host_ip()

        factory = SearchSpaceFactory()
        self._search_space = factory.get_search_space(configs)

        # create controller server
        if self._is_server:
            init_tokens = self._search_space.init_tokens()
            range_table = self._search_space.range_table()
            range_table = (len(range_table) * [0], range_table)
            _logger.info("range table: {}".format(range_table))

            if load_checkpoint != None:
                assert os.path.exists(
                    load_checkpoint
                ) == True, 'load checkpoint file NOT EXIST!!! Please check the directory of checkpoint!!!'
                checkpoint_path = os.path.join(load_checkpoint,
                                               'sanas.checkpoints')
                with open(checkpoint_path, 'r') as f:
                    scene = json.load(f)
                preinit_tokens = scene['_tokens']
                prereward = scene['_reward']
                premax_reward = scene['_max_reward']
                prebest_tokens = scene['_best_tokens']
                preiter = scene['_iter']
            else:
                preinit_tokens = init_tokens
                prereward = -1
                premax_reward = -1
                prebest_tokens = None
                preiter = 0

            controller = SAController(
                range_table,
                self._reduce_rate,
                self._init_temperature,
                max_try_times=None,
                init_tokens=preinit_tokens,
                reward=prereward,
                max_reward=premax_reward,
                iters=preiter,
                best_tokens=prebest_tokens,
                constrain_func=None,
                checkpoints=save_checkpoint)

            max_client_num = 100
            self._controller_server = ControllerServer(
                controller=controller,
                address=(server_ip, server_port),
                max_client_num=max_client_num,
                search_steps=search_steps,
                key=self._key)
            self._controller_server.start()
            server_port = self._controller_server.port()

        self._controller_client = ControllerClient(
            server_ip, server_port, key=self._key)

        if is_server and load_checkpoint != None:
            self._iter = scene['_iter']
        else:
            self._iter = 0

    def _get_host_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def tokens2arch(self, tokens):
        return self._search_space.token2arch(tokens)

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
        Returns:
            bool: True means updating successfully while false means failure.
        """
        self._iter += 1
        return self._controller_client.update(self._current_tokens, score,
                                              self._iter)

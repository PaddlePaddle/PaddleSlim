# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import time
import paddle.fluid as fluid
from ..common.RL_controller.utils import RLCONTROLLER
from ..common import get_logger
from ..analysis import flops

from ..common import Server
from ..common import Client
from .search_space import SearchSpaceFactory

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['RLNAS']


class RLNAS(object):
    """ Controller with Reinforencement Learning """

    def __init__(self,
                 key,
                 configs,
                 args,
                 server_addr=("", 8881),
                 is_server=True,
                 is_sync=False,
                 save_controller=None,
                 load_controller=None,
                 **kwargs):
        if not is_server:
            assert server_addr[
                0] != "", "You should set the IP and port of server when is_server is False."

        self._configs = configs
        factory = SearchSpaceFactory()
        self._search_space = factory.get_search_space(configs)
        self.range_tables = self._search_space.range_table()
        self.save_controller = save_controller
        self.load_controller = load_controller

        cls = RLCONTROLLER.get(key.upper())

        server_ip, server_port = server_addr
        if server_ip == None or server_ip == "":
            server_ip = self._get_host_ip()

        kwargs['range_tables'] = self.range_tables
        self._controller = cls(**kwargs)

        if is_server:
            max_client_num = 300
            self._controller_server = Server(
                controller=self._controller,
                address=(server_ip, server_port),
                is_sync=is_sync,
                save_controller=self.save_controller,
                load_controller=self.load_controller)
            self._controller_server.start()

        self._client_name = hashlib.md5(
            str(time.time() + np.random.randint(1, 10000)).encode(
                "utf-8")).hexdigest()
        self._controller_client = Client(
            controller=self._controller,
            address=(server_ip, server_port),
            client_name=self._client_name)

        self._current_tokens = None

    def _get_host_ip(self):
        try:
            return socket.gethostbyname(socket.gethostname())
        except:
            return socket.gethostbyname('localhost')

    def next_archs(self, states=None):
        """ Get next archs"""
        self._current_tokens = self._controller_client.next_tokens(states)
        archs = self._search_space.token2arch(self._current_tokens)

        return archs

    def reward(self, rewards, **kwargs):
        """ reward the score and to train controller """
        return self._controller_client.update(rewards, **kwargs)

    def final_archs(self, batch_states):
        """Get finally architecture"""
        final_tokens = self._controller_client.next_tokens(batch_states)
        archs = []
        for token in final_tokens:
            arch = self._search_space.token2arch(token)
            archs.append(arch)

        return archs

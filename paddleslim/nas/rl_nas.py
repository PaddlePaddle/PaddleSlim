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

from ..common import ControllerServer
from ..common import ControllerClient
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
                 save_controller=None,
                 load_controller=None,
                 **kwargs):
        if not is_server:
            assert server_addr[
                0] != "", "You should set the IP and port of server when is_server is False."

        factory = SearchSpaceFactory()
        self._search_space = factory.get_search_space(configs)
        self.range_tables = self._search_space.range_table()

        cls = RLCONTROLLER.get(key.upper())
        kwargs['range_tables'] = self.range_tables
        self.controller = cls(**kwargs)

        self._current_tokens = None
        self.save_controller = save_controller
        self.load_controller = load_controller

    def next_archs(self, states=None):
        """ Get next archs"""
        self._current_tokens = self.controller.next_tokens(states)
        archs = self._search_space.token2arch(self._current_tokens)

        return archs

    def reward(self, rewards, **kwargs):
        """ reward the score and to train controller """
        return self.controller.update(rewards, **kwargs)

    def final_archs(self, batch_states):
        """Get finally architecture"""
        final_tokens = self.controller.next_tokens(batch_states)
        archs = []
        for token in final_tokens:
            arch = self._search_space.token2arch(token)
            archs.append(arch)

        return archs

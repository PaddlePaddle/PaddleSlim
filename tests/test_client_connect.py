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
import sys
sys.path.append("../")
import os
import time
import signal
import unittest
from static_case import StaticCase
import paddle.fluid as fluid
from paddleslim.nas import SANAS
from paddleslim.common.controller_client import ControllerClient
import numpy as np
from multiprocessing import Process
import socket


def start_client(configs, addr, port):
    client_sanas = SANAS(
        configs=configs,
        server_addr=(addr, port),
        save_checkpoint=None,
        is_server=False)
    for _ in range(2):
        arch = client_sanas.next_archs()[0]
        time.sleep(1)
        client_sanas.reward(0.1)


def start_server(configs, port):
    server_sanas = SANAS(
        configs=configs, server_addr=("", port), save_checkpoint=None)
    server_sanas.next_archs()[0]
    return server_sanas


class TestClientConnect(StaticCase):
    def setUp(self):
        self.configs = [('MobileNetV2BlockSpace', {'block_mask': [0]})]
        self.port = np.random.randint(8337, 8773)
        self.addr = socket.gethostbyname(socket.gethostname())

    def test_client_start_first(self):
        p = Process(
            target=start_client, args=(self.configs, self.addr, self.port))
        p.start()

        start_server(self.configs, self.port)


class TestClientConnectCase1(StaticCase):
    def setUp(self):
        self.configs = [('MobileNetV2BlockSpace', {'block_mask': [0]})]
        self.port = np.random.randint(8337, 8773)
        self.addr = socket.gethostbyname(socket.gethostname())

    def test_client_start_first(self):
        p = Process(
            target=start_client, args=(self.configs, self.addr, self.port))
        p.start()

        time.sleep(60)
        server_sanas = start_server(self.configs, self.port)
        os.kill(os.getpid(), 0)


class TestClientConnectCase2(StaticCase):
    def setUp(self):
        self.port = np.random.randint(8337, 8773)
        self.addr = socket.gethostbyname(socket.gethostname())

    def test_request_current_info(self):
        client = ControllerClient(self.addr, self.port)
        client.request_current_info()


if __name__ == '__main__':
    unittest.main()

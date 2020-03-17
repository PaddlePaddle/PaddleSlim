import os
import six
import time
import signal
if six.PY2:
    import Queue
else:
    import queue as Queue
import logging
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import paddle.fluid as fluid
from .log_helper import get_logger

__all__ = ['Client']

_logger = get_logger(__name__, level=logging.INFO)
PublicAuthKey = u'AbcXyz3'


class Client(object):
    def __init__(self, controller, address, client_name):
        self._controller = controller
        self._address = address
        self._client_name = client_name
        self._manager = self._start_manager()
        self._client = {self._client_name: -1}
        try:
            client_dict = self._manager.get_client_dict()
            client_dict.update(self._client)
        except Exception as err:
            _logger.error("get information from server error: {}".format(err))
            pid = os.getpid()
            os.kill(pid, signal.SIGKILL)

    def _start_manager(self):
        BaseManager.register('get_client_queue')
        BaseManager.register('get_params_dict')
        BaseManager.register('get_client_dict')
        BaseManager.register('get_lock')

        manager = BaseManager(
            address=self._address, authkey=PublicAuthKey.encode())

        while True:
            try:
                manager.connect()
                break
            except:
                time.sleep(0.1)
        return manager

    def list2dict(self, lists):
        res_dict = dict()
        for l in lists:
            tmp_dict = dict()
            tmp_dict[l[0]] = l[1]
            res_dict.update(tmp_dict)
        return res_dict

    def next_tokens(self, status):
        lock = self._manager.get_lock()
        lock.acquire()
        print('next_tokens: start lock', self._client_name,
              time.asctime(time.localtime(time.time())), lock)
        params_dict = self._manager.get_params_dict()
        while len(params_dict.items()) == 0:
            params_dict = self._manager.get_params_dict()
        lock.release()
        print("next_tokens: lock release", self._client_name,
              time.asctime(time.localtime(time.time())), lock)
        params_dict = self.list2dict(params_dict.items())
        tokens = self._controller.next_tokens(status, params_dict)

        return tokens

    def update(self, rewards, **kwargs):
        print(rewards)
        self._client[self._client_name] = rewards
        client_dict = self._manager.get_client_dict()
        client_dict.update(self._client)
        lock = self._manager.get_lock()
        print("wait ...", lock)
        lock.acquire()
        print('update: start lock', self._client_name,
              time.asctime(time.localtime(time.time())), lock)

        params_dict = self._manager.get_params_dict()
        while len(params_dict.items()) == 0:
            params_dict = self._manager.get_params_dict()
        np_params_dict = self.list2dict(params_dict.items())

        try:
            current_params_dict = self._controller.update(
                rewards, np_params_dict, **kwargs)
        except Exception as err:
            lock.release()
            _logger.error("update: get parameter from server error: {}".format(
                err))
            pid = os.getpid()
            os.kill(pid, signal.SIGKILL)

        params_dict = current_params_dict
        lock.release()
        print("update: lock release", self._client_name,
              time.asctime(time.localtime(time.time())), lock)

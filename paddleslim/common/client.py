import os
import six
import signal
if six.PY2:
    import Queue
else:
    import queue as Queue
import logging
import numpy as np
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
        while True:
            try:
                client_dict = self._manager.get_client_dict()
                client_dict.update(self._client)
                client_queue = self._manager.get_client_queue()
                client_queue.put(self._client_name)
                break
            except Exception as err:
                _logger.error("get information from server error: {}".format(
                    err))
                pid = os.getpid()
                os.kill(pid, signal.SIGKILL)

    def _start_manager(self):
        BaseManager.register('get_client_queue')
        BaseManager.register('get_params_dict')
        BaseManager.register('get_client_dict')

        manager = BaseManager(
            address=self._address, authkey=PublicAuthKey.encode())
        manager.connect()
        return manager

    def list2dict(self, lists):
        res_dict = dict()
        for l in lists:
            tmp_dict = dict()
            tmp_dict[l[0]] = l[1]
            res_dict.update(tmp_dict)
        return res_dict

    def next_tokens(self, status):
        while True:
            try:
                params_dict = self._manager.get_params_dict()
                if len(params_dict.keys()) == 0:
                    time.sleep(1)
                else:
                    break
            except Exception as err:
                _logger.error(
                    "next_tokens: get parameter from server error: {}".format(
                        err))
                pid = os.getpid()
                os.kill(pid, signal.SIGKILL)
        params_dict = self.list2dict(params_dict.items())
        tokens = self._controller.next_tokens(status, params_dict)

    def update(self, rewards, **kwargs):
        self._client[self._client_name] = rewards
        while True:
            try:
                client_dict = self._manager.get_client_dict()
                client_dict.update(self._client)
                params_dict = self._manager.get_params_dict()
                if len(params_dict.keys()) == 0:
                    time.sleep(1)
                else:
                    break
            except Exception as err:
                _logger.error(
                    "update: get parameter from server error: {}".format(err))
                pid = os.getpid()
                os.kill(pid, signal.SIGKILL)

        params_dict = self.list2dict(params_dict.items())
        self._controller.update(rewards, params_dict, **kwargs)

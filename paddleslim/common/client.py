import os
import six
import time
import signal
import logging
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
    def __init__(self, controller, address, client_name, is_sync=False):
        self._controller = controller
        self._address = address
        self._client_name = client_name
        self._update_times = 0
        self._is_sync = is_sync
        self._manager = self._start_manager()
        try:
            client_list = self._manager.get_client_list()
            client_list.put(self._client_name)
        except Exception as err:
            _logger.error("get information from server error: {}".format(err))
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)

    def _start_manager(self):
        BaseManager.register('get_client_queue')
        BaseManager.register('get_params_dict')
        BaseManager.register('get_client_list')
        BaseManager.register('get_update_lock')
        BaseManager.register('get_max_update_times')

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
        lock = None
        if self._is_sync and self._update_times != 0:
            ### sync mode, next token must wait all client update params if this client is not added now.
            queue_size = self._manager.get_client_queue().qsize()
            client_size = self._manager.get_client_list().qsize()
            while queue_size < client_size:
                queue_size = self._manager.get_client_queue().qsize()
                client_size = self._manager.get_client_list().qsize()
        else:
            ### async mode, next token can get params if params is not write now
            ### sync mode, if client is added just now and params is writing now, need to wait all client write done
            lock = self._manager.get_update_lock()
        if lock != None:
            lock.acquire()
        params_dict = self._manager.get_params_dict()
        while len(params_dict.items()) == 0:
            params_dict = self._manager.get_params_dict()

        max_update_times = self._manager.get_max_update_times().get()
        if max_update_times > self._update_times:
            self._update_times = max_update_times
            self._manager.get_max_update_times().put(self._update_times)
        else:
            self._manager.get_max_update_times().put(max_update_times)
        if lock != None:
            lock.release()
        self.params_dict = self.list2dict(params_dict.items())
        tokens = self._controller.next_tokens(status, self.params_dict)

        return tokens

    def update(self, rewards, **kwargs):
        self._update_times += 1

        if not self._is_sync:
            lock = self._manager.get_update_lock()
            lock.acquire()
            max_update_times_queue = self._manager.get_max_update_times()
            max_update_times = max_update_times_queue.get()
            ### in async mode, if current client current is behind than current params, drop this update
            if self._update_times < max_update_times:
                max_update_times_queue.put(max_update_times)
                lock.release()
                return

            max_update_times_queue.put(self._update_times)
            lock.release()

        try:
            current_params_dict = self._controller.update(
                rewards, self.params_dict, **kwargs)
        except Exception as err:
            lock.release()
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)

        ### compute mean update
        if self._is_sync:
            client_num = self._manager.get_client_list().qsize()
            for key, value in current_params_dict.items():
                current_params_dict[key] -= self.params_dict[key]
                current_params_dict[key] = list(
                    map(lambda x: x / np.array(client_num).astype(np.float32),
                        value))

        lock = self._manager.get_update_lock()
        lock.acquire()

        params_dict = self._manager.get_params_dict()
        if self._is_sync:
            params_dict_ = self.list2dict(params_dict.items())
            for key, value in params_dict_.items():
                current_params_dict[key] += params_dict_[key]

        params_dict.update(current_params_dict)

        ### used in sync mode to controll client get next token
        queue_size = self._manager.get_client_queue().qsize()
        if queue_size == self._manager.get_client_list().qsize():
            while queue_size > 0:
                self._manager.get_client_queue().get()
                queue_size = self._manager.get_client_queue().qsize()

        client_queue = self._manager.get_client_queue()
        client_queue.put(self._client_name)
        lock.release()

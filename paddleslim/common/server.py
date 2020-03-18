import os
import six
if six.PY2:
    import Queue
else:
    import queue as Queue
import logging
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import RLock, Process
from multiprocessing.managers import BaseManager
from ctypes import c_bool, c_ulong
from threading import Thread
import paddle.fluid as fluid
from .log_helper import get_logger

__all__ = ['Server']

_logger = get_logger(__name__, level=logging.INFO)
PublicAuthKey = u'AbcXyz3'

client_queue = Queue.Queue(300)
update_lock = RLock()
max_update_times = Queue.Queue(1)
client_list = Queue.Queue()
params_dict = dict()


class Server(object):
    def __init__(self,
                 controller,
                 address,
                 is_sync=False,
                 load_controller=None,
                 save_controller=None,
                 args=None):
        self._controller = controller
        self._address = address
        self._args = args
        self._save_controller = save_controller
        self._load_controller = load_controller
        self._is_sync = is_sync

    def _start_manager(self):
        def get_client_queue():
            global client_queue
            return client_queue

        def get_params_dict():
            global params_dict
            return params_dict

        def get_client_list():
            global client_list
            return client_list

        def get_update_lock():
            global update_lock
            return update_lock

        def get_max_update_times():
            global max_update_times
            return max_update_times

        BaseManager.register('get_client_queue', callable=get_client_queue)
        BaseManager.register('get_params_dict', callable=get_params_dict)
        BaseManager.register('get_client_list', callable=get_client_list)
        BaseManager.register(
            'get_max_update_times', callable=get_max_update_times)
        BaseManager.register('get_update_lock', callable=get_update_lock)
        manager = BaseManager(
            address=self._address, authkey=PublicAuthKey.encode())
        manager.start()
        return manager

    def start(self):
        self._manager = self._start_manager()
        self._params_dict = self._manager.get_params_dict()
        max_update_times = self._manager.get_max_update_times()
        max_update_times.put(0)

        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self._controller._build_program(self.main_program,
                                        self.startup_program)

        self.place = fluid.CUDAPlace(
            0)  # if self.args.controller_use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(self.place)
        exe.run(self.startup_program)

        if self._load_controller:
            assert os.path.exists(
                self._load_controller
            ), "controller checkpoint is not exist, please check your directory: {}".format(
                self._load_controller)
            self._controller.load_controller(self.main_program,
                                             self._load_controller)

        var_dict = self._controller.get_params(self.main_program)

        self._params_dict.update(var_dict)

        listen = Thread(target=self._save_params, args=())
        listen.setDaemon(True)
        listen.start()

    def _save_params(self):
        while True:
            if int(time.time()) % 3600 == 0:
                params_dict = self._manager.get_params_dict()

                def list2dict(lists):
                    res_dict = dict()
                    for l in lists:
                        tmp_dict = dict()
                        tmp_dict[l[0]] = l[1]
                        res_dict.update(tmp_dict)
                    return res_dict

                params_dict = list2dict(self._params_dict.items())

                self._controller.set_params(self.main_program, params_dict,
                                            self.place)

                if self._save_controller:
                    self._controller.save_controller(self.main_program,
                                                     self._save_controller)

    def __del__(self):
        if self._manager:
            self._manager.shutdown()

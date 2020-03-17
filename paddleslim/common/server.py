#collection parameters + Lock
#
#from multiprocessing import Lock

import os
import six
if six.PY2:
    import Queue
else:
    import queue as Queue
import logging
import numpy as np
import atexit
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import paddle.fluid as fluid
from .log_helper import get_logger

__all__ = ['Server']

_logger = get_logger(__name__, level=logging.INFO)
PublicAuthKey = u'AbcXyz3'

client_queue = Queue.Queue(300)
current_client = list()
client_dict = dict()
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
        #self._client_queue = Queue.Queue(300)
        #self.lock = mp.Lock()
        #atexit.register(self.before_exit)

    def _start_manager(self):
        def get_client_queue():
            global client_queue
            return client_queue

        def get_params_dict():
            global params_dict
            return params_dict

        def get_client_dict():
            global client_dict
            return client_dict

        def get_current_client():
            global current_client
            return current_client

        BaseManager.register('get_client_queue', callable=get_client_queue)
        BaseManager.register('get_params_dict', callable=get_params_dict)
        BaseManager.register('get_client_dict', callable=get_client_dict)
        BaseManager.register('get_current_client', callable=get_current_client)
        manager = BaseManager(
            address=self._address, authkey=PublicAuthKey.encode())
        manager.start()
        return manager

    def start(self):
        self._manager = self._start_manager()
        self._params_dict = self._manager.get_params_dict()

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
                                             self.load_controller)

        var_dict = dict()
        for var in self.main_program.global_block().all_parameters():
            var_dict[var.name] = np.array(fluid.global_scope().find_var(
                var.name).get_tensor())

        self._params_dict.update(var_dict)

        #self.run_sync()
        #thread = Thread(target = )
        #thread.setDaemon(True)
        #thread.start()

    def run_sync(self):
        pass

    def run_async(self):
        try:
            self._client_queue = self._manager.get_client_queue()
            self._current_client = self._manager.get_current_client()
            self._current_client.clear()
            if not self._client_queue.empty():
                self._current_client = self._client_queue.get()
        except:
            pass

    def before_exit(self):
        params_dict = self._manager.get_params_dict()

        def list2dict(lists):
            res_dict = dict()
            for l in lists:
                tmp_dict = dict()
                tmp_dict[l[0]] = l[1]
                res_dict.update(tmp_dict)
            return res_dict

        params_dict = list2dict(self._params_dict.items())
        #print(params_dict)

        for var in self.main_program.global_block().all_parameters():
            fluid.global_scope().find_var(var.name).get_tensor().set(
                params_dict[var.name], self.place)

        if self._save_controller:
            self._controller.save_controller(self.main_program,
                                             self._save_controller)

    def __del__(self):
        #self.before_exit()
        if self._manager:
            self._manager.shutdown()

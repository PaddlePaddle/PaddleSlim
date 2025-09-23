#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""utils for ViT

Contains AverageMeter for monitoring, get_exclude_from_decay_fn for training
and WarmupCosineScheduler for training

"""

import logging
import sys
import os
import paddle
import paddle.distributed as dist


def get_logger(file_path):
    """Set logging file and format, logs are written in 2 loggers, one local_logger records
       the information on its own gpu/process, one master_logger records the overall/average
       information over all gpus/processes.
    Args:
        file_path: str, folder path of the logger files to write
    Return:
        local_logger: python logger for each process
        master_logger: python logger for overall processes (on node 0)
    """
    local_rank = dist.get_rank()
    filename = os.path.join(file_path, 'log_all.txt')
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(filename=filename, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")

    # local_logger for each process/GPU
    local_logger = logging.getLogger(f'local_{local_rank}')
    filename = os.path.join(file_path, f'log_{local_rank}.txt')
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(log_format))
    local_logger.addHandler(fh)
    ## console
    #sh = logging.StreamHandler(sys.stdout)
    #sh.setFormatter(logging.Formatter(log_format))
    #local_logger.addHandler(sh)

    # master_logger records avg performance
    if local_rank == 0:
        master_logger = logging.getLogger('master')
        # log.txt
        filename = os.path.join(file_path, 'log.txt')
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(fh)
        # console
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(sh)
    else:
        master_logger = None
    return local_logger, master_logger


def write_log(local_logger, master_logger, msg_local, msg_master=None, level='info'):
    """Write messages in loggers
    Args:
        local_logger: python logger, logs information on single gpu
        master_logger: python logger, logs information over all gpus
        msg_local: str, message to log on local_logger
        msg_master: str, message to log on master_logger, if None, use msg_local, default: None
        level: str, log level, in ['info', 'warning', 'fatal'], default: 'info'
    """
    # write log to local logger
    if local_logger:
        if level == 'info':
            local_logger.info(msg_local)
        elif level == 'warning':
            local_logger.warning(msg_local)
        elif level == 'fatal':
            local_logger.fatal(msg_local)
        else:
            raise ValueError("level must in ['info', 'warning', 'fatal']")
    # write log to master logger on node 0
    if master_logger and dist.get_rank() == 0:
        if msg_master is None:
            msg_master = msg_local
        if level == 'info':
            master_logger.info("MASTER_LOG " + msg_master)
        elif level == 'warning':
            master_logger.warning("MASTER_LOG " + msg_master)
        elif level == 'fatal':
            master_logger.fatal("MASTER_LOG " + msg_master)
        else:
            raise ValueError("level must in ['info', 'warning', 'fatal']")


def all_reduce_mean(x):
    """perform all_reduce on Tensor for gathering results from multi-gpus"""
    world_size = dist.get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce = x_reduce / world_size
        return x_reduce.item()
    return x


class AverageMeter():
    """ Meter for monitoring losses"""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

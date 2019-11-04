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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
from six.moves.queue import Queue

import paddle.fluid as fluid
from paddle.fluid.framework import Variable
from paddle.fluid.reader import DataLoaderBase
from paddle.fluid.core import EOFException
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ['Knowledge']


class Knowledge(object):
    """
    The knowledge class describes how to extract and store the dark knowledge
    of the teacher model, and how the student model learns these dark knowledge.
    """

    def __init__(self,
                 path,
                 items,
                 reduce_strategy={'type': 'sum',
                                  'key': 'image'}):
        """Init a knowledge instance.
        Args:
            path(list<str>, str, optional): Specifies the storage path of the knowledge,
                       supports AFS/HDFS, local file system, and memory.
            items(list<str>): Save the tensor of the specified name
            reduce_strategy(dict, optional): The policy for performing the reduce
                                   operation. If it is set to None,
                                   the reduce operation is not performed.
            reduce_strategy.type(str): Type of reduce operation.
            reduce_strategy.key(str): The key of the reduce operation.
                                      It is an element in the item.
        """
        assert (isinstance(path, list) or isinstance(path, str) or
                (path is None)), "path type should be list or str or None"
        assert (isinstance(items, list)), "items should be a list"
        assert (isinstance(reduce_strategy,
                           dict)), "reduce_strategy should be a dict"
        self.path = path
        if isinstance(self.path, list):
            self.write_type = 'HDFS/AFS'
            assert (
                len(self.path) == 4 and isinstance(self.path[0], str) and
                isinstance(self.path[1], str) and
                isinstance(self.path[2], str) and isinstance(self.path[3], str)
            ), "path should contains four str, ['local hadoop home', 'fs.default.name', 'hadoop.job.ugi', 'FS path']"

            hadoop_home = self.path[0]
            configs = {
                "fs.default.name": self.path[1],
                "hadoop.job.ugi": self.path[2]
            }
            self.client = HDFSClient(hadoop_home, configs)
            assert (
                self.client.is_exist(self.path[3]) == True
            ), "Plese make sure your hadoop confiuration is correct and FS path exists"

            self.hdfs_local_path = "./teacher_knowledge"
            if not os.path.exists(self.hdfs_local_path):
                os.mkdir(self.hdfs_local_path)
        elif isinstance(self.path, str):
            self.write_type = "LocalFS"
            if not os.path.exists(path):
                raise ValueError("The local path [%s] does not exist." %
                                 (path))
        else:
            self.write_type = "MEM"
            self.knowledge_queue = Queue(64)

        self.items = items
        self.reduce_strategy = reduce_strategy

    def _write(self, data):
        if self.write_type == 'HDFS/AFS':
            file_name = 'knowledge_' + str(self.file_cnt)
            file_path = os.path.join(self.hdfs_local_path, file_name)
            file_path += ".npy"
            np.save(file_path, data)
            self.file_cnt += 1
            self.client.upload(self.path[3], file_path)
            logger.info('{}.npy pushed to HDFS/AFS: {}'.format(file_name,
                                                               self.path[3]))

        elif self.write_type == 'LocalFS':
            file_name = 'knowledge_' + str(self.file_cnt)
            file_path = os.path.join(self.path, file_name)
            np.save(file_path, data)
            logger.info('{}.npy saved'.format(file_name))
            self.file_cnt += 1

        else:
            self.knowledge_queue.put(data)
            logger.info('{} pushed to Queue'.format(file_name))

    def run(self, teacher_program, exe, place, scope, reader, inputs, outputs,
            call_back):
        """Start teacher model to do information.
        Args:
            teacher_program(Program): teacher program.
            scope(Scope): The scope used to execute the teacher,
                          which contains the initialized variables.
            reader(reader): The data reader used by the teacher.
            inputs(list<str>): The name of variables to feed the teacher program.
            outputs(list<str>): Need to write to the variable instance's names of
                                the Knowledge instance, which needs to correspond
                                to the Knowledge's items.
            call_back(func, optional): The callback function that handles the
                          outputs of the teacher, which is none by default,
                          that is, the output of the teacher is concat directly.
        Return:
            (bool): Whether the teacher task was successfully registered and started
        """
        assert (isinstance(
            teacher_program,
            fluid.Program)), "teacher_program should be a fluid.Program"
        assert (isinstance(inputs, list)), "inputs should be a list"
        assert (isinstance(outputs, list)), "outputs should be a list"
        assert (len(self.items) == len(outputs)
                ), "the length of outputs list should be equal with items list"
        assert (callable(call_back) or (call_back is None)
                ), "call_back should be a callable function or NoneType."

        for var in teacher_program.list_vars():
            var.stop_gradient = True

        compiled_teacher_program = fluid.compiler.CompiledProgram(
            teacher_program)
        self.file_cnt = 0
        if isinstance(reader, Variable) or (
                isinstance(reader, DataLoaderBase) and (not reader.iterable)):
            reader.start()
            try:
                while True:
                    logits = exe.run(compiled_teacher_program,
                                     scope=scope,
                                     fetch_list=outputs,
                                     feed=None)
                    knowledge = dict()
                    for index, array in enumerate(logits):
                        knowledge[self.items[index]] = array
                    self._write(knowledge)
            except EOFException:
                reader.reset()

        else:
            if not isinstance(reader, DataLoaderBase):
                feeder = fluid.DataFeeder(
                    feed_list=inputs, place=place, program=teacher_program)
            for batch_id, data in enumerate(reader()):
                if not isinstance(reader, DataLoaderBase):
                    data = feeder.feed(data)
                logits = exe.run(compiled_teacher_program,
                                 scope=scope,
                                 fetch_list=outputs,
                                 feed=data)
                knowledge = dict()
                for index, array in enumerate(logits):
                    knowledge[self.items[index]] = array
                self._write(knowledge)
        return True

    def dist(self, student_program, losses):
        """Building the distillation network
        Args:
            student_program(Program): student program.
            losses(list<Variable>, optional): The losses need to add. If set to None
                              does not add any loss.
        Return:
            (Program): Program for distillation.
            (startup_program): Program for initializing distillation network.
            (reader): Data reader for distillation training.
            (Variable): Loss of distillation training
        """

    def loss(self, loss_func, *variables):
        """User-defined loss
        Args:
            loss_func(func): Function used to define loss.
            *variables(list<str>): Variable name list.
        Return:
            (Variable): Distillation loss.
        """
        pass

    def fsp_loss(self):
        """fsp loss
        """
        pass

    def l2_loss(self):
        """l2 loss
        """
        pass

    def softlabel_loss(self):
        """softlabel_loss
        """
        pass

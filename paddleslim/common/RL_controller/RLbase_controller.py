#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
"""The controller used to search hyperparameters or neural architecture"""

import os
import sys
import logging
import paddle.fluid as fluid
from ..controller import EvolutionaryController
from ..log_helper import get_logger

__all__ = ['RLBaseController']

_logger = get_logger(__name__, level=logging.INFO)


class RLBaseController(EvolutionaryController):
    """ Base Controller for reforcement learning"""

    def init(self, args):
        """ initial parameter in reforcement learning network"""
        self.args = args

    def _create_input(self, **kwargs):
        raise NotImplementedError('Abstract method.')

    def _build_program(self, *args, **kwargs):
        raise NotImplementedError('Abstract method.')

    def next_tokens(self, num_archs=1):
        """ sample next tokens according current parameter and inputs"""
        main_program = fluid.Program()
        startup_program = fluid.Program()
        inputs, loss = self._build_program(
            main_program, startup_program, is_test=True, batch_size=batch_size)

        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        build_strategy = fluid.BuildStrategy()
        compiled_program = fluid.CompiledProgram(
            main_program).with_data_parallel(
                loss.name, build_strategy=build_strategy)
        feed_dict = self._create_input(inputs)

        token = exe.run(compiled_program, feed=feed_dict, fetch_list=[tokens])
        return token

    def update(self, tokens, reward):
        """train controller according reward"""
        main_program = fluid.Program()
        startup_program = fluid.Program()
        inputs, loss = self._build_program(main_program, startup_program)

        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        feed_dict = self._create_input(
            inputs, is_test=False, actual_rewards=reward)

        build_strategy = fluid.BuildStrategy()
        compiled_program = fluid.CompiledProgram(
            main_program).with_data_parallel(
                loss.name, build_strategy=build_strategy)

        token = exe.run(compiled_program, feed=feed_dict, fetch_list=[tokens])
        if token == self.token:
            return False
        else:
            #    if self.save_controller is not None:
            #        self._save_controller(main_program)
            return True

    def _save_controller(self, program):
        fluid.save(program, self.save_controller)

    def record(self):
        """ record information needed in reforcement learning."""
        raise NotImplementedError('Abstract method.')

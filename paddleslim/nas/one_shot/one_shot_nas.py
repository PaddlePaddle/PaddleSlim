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

import numpy as np
import paddle.fluid as fluid
from ...common import SAController

__all__ = ['OneShotSuperNet', 'OneShotSearch']


def OneShotSearch(model, eval_func, strategy='sa', search_steps=100):
    """
    Search a best tokens which represents a sub-network.

    Args:
        model(fluid.dygraph.Layer): A dynamic graph module whose sub-modules should contain
                                    one instance of `OneShotSuperNet` at least.
        eval_func(function): A callback function which accept model and tokens as arguments.
        strategy(str): The name of strategy used to search. Default: 'sa'.
        search_steps(int): The total steps for searching.

    Returns:
        list<int>: The best tokens searched.
    """
    super_net = None
    for layer in model.sublayers(include_self=True):
        print("layer: {}".format(layer))
        if isinstance(layer, OneShotSuperNet):
            super_net = layer
            break
    assert super_net is not None
    controller = None
    if strategy == "sa":
        contoller = SAController(
            range_table=super_net.range_table(),
            init_tokens=super_net.init_tokens())
    assert (controller is not None, "Unsupported searching strategy.")
    for i in range(search_steps):
        tokens = contoller.next_tokens()
        reward = eval_func(model, tokens)
        contoller.update(tokens, reward, i)
    return contoller.best_tokens()


class OneShotSuperNet(fluid.dygraph.Layer):
    """The base class of super net used in one-shot searching strategy.
    A super net is a dygraph layer.
    
    Args:
        name_scope(str): The name scope of super net.
    """

    def __init__(self, name_scope):
        super(OneShotSuperNet, self).__init__(name_scope)

    def init_tokens(self):
        """Get init tokens in search space.

        Returns:
           lis<int>t: The init tokens which is a list of integer.
        """
        raise NotImplementedError('Abstract method.')

    def range_table(self):
        """Get range table of current search space.

        Returns:
           range_table(tuple): The maximum value and minimum value in each position of tokens
                               with format `(min_values, max_values)`. The `min_values` is
                               a list of integers  indicating the minimum values while `max_values`
                               indicating the maximum values.
        """
        raise NotImplementedError('Abstract method.')

    def _forward_impl(self, *inputs, **kwargs):
        """Defines the computation performed at every call.
        Should be overridden by all subclasses.

        Args:
            inputs(tuple): unpacked tuple arguments
            kwargs(dict): unpacked dict arguments
        """
        raise NotImplementedError('Abstract method.')

    def forward(self, input, tokens=None):
        """
        Defines the computation performed at every call.

        Args:
            input(variable): The input of super net.
            tokens(list): The tokens used to generate a sub-network.
                          None means computing in super net training mode.
                          Otherwise, it will execute the sub-network generated by tokens.
                          The `tokens` should be set in searching stage and final training stage.
                          Default: None.

        Returns:
            Varaible: The output of super net.
        """
        if tokens == None:
            tokens = self._random_tokens()
        return self._forward_impl(input, tokens=tokens)

    def _random_tokens(self):
        tokens = []
        for min_v, max_v in zip(self.range_table()[0], self.range_table()[1]):
            tokens.append(np.random.randint(min_v, max_v))
        return tokens

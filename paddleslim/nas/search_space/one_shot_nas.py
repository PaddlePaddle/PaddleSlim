import numpy as np
import paddle.fluid as fluid
from .search_space_registry import SEARCHSPACE
from .super_mnasnet import SuperMnasnet
from .search_space_base import SearchSpaceBase

__all__ = ['OneShotMnasnetSearchSpace']


@SEARCHSPACE.register
class OneShotMnasnetSearchSpace(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        self.name_scope = "mnasnet"
        self.input_size = input_size
        self.output_size = output_size

    def init_tokens(self):
        return [
            3, 3, 6, 6, 6, 6, 3, 3, 3, 6, 6, 6, 3, 3, 3, 3, 6, 6, 3, 3, 3, 6,
            6, 6, 3, 3, 3, 6, 6, 6, 3, 6, 6, 6, 6, 6
        ]

    def range_table(self):
        return [
            6, 6, 10, 10, 10, 10, 6, 6, 6, 10, 10, 10, 6, 6, 6, 6, 10, 10, 6,
            6, 6, 10, 10, 10, 6, 6, 6, 10, 10, 10, 6, 10, 10, 10, 10, 10
        ]
        #return [5] * 36

    def token2arch(self, tokens=None):
        return SubNet(
            self.range_table,
            self.name_scope,
            input_channels=self.input_size,
            output_channels=self.output_size,
            tokens=tokens)

    def super_net(self):
        net = SubNet(
            self.range_table,
            self.name_scope,
            input_channels=self.input_size,
            output_channels=self.output_size,
            tokens=None)
        net.super_mode = True
        return net


class SubNet(fluid.dygraph.Layer):
    def __init__(self,
                 range_table,
                 name_scope,
                 input_channels=None,
                 output_channels=None,
                 tokens=None):
        super(SubNet, self).__init__(name_scope)
        self.tokens = tokens
        self.range_table = range_table
        self.super_mode = False
        if self.tokens is None:
            self.super_mode = True
        self.super_net = SuperMnasnet(
            name_scope,
            input_channels=input_channels,
            out_channels=output_channels)

    def forward(self, input):
        if self.super_mode == True:
            tokens = self.random_tokens()
            return self.super_net(input, tokens)
        else:
            return self.super_net(input, self.tokens)

    def random_tokens(self):
        tokens = []
        for n_range in self.range_table():
            tokens.append(np.random.randint(n_range))
        return tokens

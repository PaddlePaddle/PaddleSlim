import numpy as np
import paddle.fluid as fluid
from ...common import SAController

__all__ = ['OneShotSuperNet', 'OneShotSearch']


def OneShotSearch(model, eval_func):
    super_net = None
    for layer in model.sublayers(include_sublayers=False):
        print("layer: {}".format(layer))
        if isinstance(layer, OneShotSuperNet):
            super_net = layer
            break
    assert super_net is not None

    contoller = SAController(
        range_table=super_net.range_table(),
        init_tokens=super_net.init_tokens())
    for i in range(100):
        tokens = contoller.next_tokens()
        reward = eval_func(model, tokens)
        contoller.update(tokens, reward, i)
    return contoller.best_tokens()


class OneShotSuperNet(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(OneShotSuperNet, self).__init__(name_scope)

    def init_tokens(self):
        raise NotImplementedError('Abstract method.')

    def range_table(self):
        raise NotImplementedError('Abstract method.')

    def forward_impl(self, *inputs, **kwargs):
        raise NotImplementedError('Abstract method.')

    def forward(self, input, tokens=None):
        if tokens == None:
            tokens = self.random_tokens()
        return self.forward_impl(input, tokens=tokens)

    def random_tokens(self):
        tokens = []
        for min_v, max_v in zip(self.range_table()[0], self.range_table()[1]):
            tokens.append(np.random.randint(min_v, max_v))
        return tokens

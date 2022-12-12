import paddle.nn as nn
import math


class MyNetwork(nn.Layer):
    def forward(self, x):
        raise NotImplementedError

    def feature_extract(self, x):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    def cfg2params(self, cfg):
        raise NotImplementedError

    def cfg2flops(self, cfg):
        raise NotImplementedError

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self):
        return self.get_parameters()

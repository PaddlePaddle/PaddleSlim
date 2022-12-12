import paddle.nn as nn


class MyNetwork(nn.Layer):
    def feature_extract(self, x):
        raise NotImplementedError

    def cfg2flops(self, cfg):
        raise NotImplementedError

import unittest
import paddle
import sys
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(path))

from paddleslim.nas.itpruner import ITPruner
from paddleslim.nas.itpruner.Cifar.nets.resnet_cifar import ResNetCifar


class TestITPruner(unittest.TestCase):
    def test_itpruner(self):
        net = ResNetCifar(depth=20, num_classes=10, cfg=None)
        data = paddle.normal(shape=[100, 3, 32, 32])

        itpruner = ITPruner(net, data)
        target_flops = 20800000
        beta = 243

        itpruner.prune(target_flops, beta)


if __name__ == '__main__':
    unittest.main()



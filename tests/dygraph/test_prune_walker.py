import sys
sys.path.append("../../")
import unittest
import numpy as np
import paddle
from paddleslim.dygraph import L1NormFilterPruner
from paddle.nn import Conv2D, Linear, Layer


class Net(Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(3, 8, 3)
        self.linear = Linear(8 * 30 * 30, 5)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = paddle.flatten(tmp, 1)
        return self.linear(tmp)


class TestWalker(unittest.TestCase):
    def runTest(self):
        x_shape = (1, 3, 32, 32)
        net = Net()
        x = np.random.uniform(-1, 1, x_shape).astype('float32')
        pruner = L1NormFilterPruner(net, [paddle.to_tensor(x)])
        pruner.prune_vars({"conv2d_0.w_0": 0.2}, [0])
        self.assertTrue(net.linear.weight.shape == [5400, 5])


if __name__ == '__main__':
    unittest.main()

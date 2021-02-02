import sys
sys.path.append("../../")
import unittest
import numpy as np
import paddle
from paddleslim import flops
from paddle.vision.models import mobilenet_v1, resnet50
from paddle.nn import Conv2D, Layer


class TestFlops(unittest.TestCase):
    def __init__(self, methodName='runTest', net=None, gt=None):
        super(TestFlops, self).__init__(methodName)
        self._net = net
        self._gt = gt

    def runTest(self):
        net = self._net(pretrained=False)
        FLOPs = flops(net, (1, 3, 32, 32), only_conv=False)
        self.assertTrue(FLOPs == self._gt)


class Net1(Layer):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = Conv2D(3, 2, 3)
        self.conv2 = Conv2D(3, 2, 3)

    def forward(self, inputs):
        assert isinstance(inputs, dict)
        x = inputs["x"]
        y = inputs["y"]
        return {"x": self.conv1(x), "y": self.conv2(y), "dummy": "dummy"}


class Net2(Net1):
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, x, y):
        return [self.conv1(x), self.conv2(y), "dummy"]


class TestFLOPsCase1(unittest.TestCase):
    def runTest(self):
        x_shape = (1, 3, 32, 32)
        y_shape = (1, 3, 16, 16)
        net = Net1()
        x = np.random.uniform(-1, 1, x_shape).astype('float32')
        y = np.random.uniform(-1, 1, y_shape).astype('float32')

        inputs = {
            "x": paddle.to_tensor(x),
            "y": paddle.to_tensor(y),
            "z": "test"
        }
        FLOPs = flops(net, [inputs], only_conv=False)
        self.assertTrue(FLOPs == 59184)


class TestFLOPsCase2(unittest.TestCase):
    def runTest(self):
        x_shape = (1, 3, 32, 32)
        y_shape = (1, 3, 16, 16)
        net = Net2()
        x = np.random.uniform(-1, 1, x_shape).astype('float32')
        y = np.random.uniform(-1, 1, y_shape).astype('float32')

        inputs = [paddle.to_tensor(x), paddle.to_tensor(y)]
        FLOPs1 = flops(net, inputs, only_conv=False)
        shapes = [x_shape, y_shape]
        FLOPs2 = flops(
            net, shapes, dtypes=["float32", "float32"], only_conv=False)
        self.assertTrue(FLOPs1 == FLOPs2)


def add_cases(suite):
    suite.addTest(TestFlops(net=mobilenet_v1, gt=11792896.0))
    suite.addTest(TestFlops(net=resnet50, gt=83872768.0))
    suite.addTest(TestFLOPsCase1())
    suite.addTest(TestFLOPsCase2())


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()

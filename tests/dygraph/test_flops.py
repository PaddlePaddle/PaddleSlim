import sys
sys.path.append("../../")
import unittest
from paddleslim.analysis import dygraph_flops as flops
from paddle.vision.models import mobilenet_v1, resnet50


class TestFlops(unittest.TestCase):
    def __init__(self, methodName='runTest', net=None, gt=None):
        super(TestFlops, self).__init__(methodName)
        self._net = net
        self._gt = gt

    def runTest(self):
        net = self._net(pretrained=False)
        FLOPs = flops(net, (1, 3, 32, 32))
        self.assertTrue(FLOPs == self._gt)


def add_cases(suite):
    suite.addTest(TestFlops(net=mobilenet_v1, gt=11792896.0))
    suite.addTest(TestFlops(net=resnet50, gt=83872768.0))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()

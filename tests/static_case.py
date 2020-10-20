import unittest
import paddle


class StaticCase(unittest.TestCase):
    def __init__(self, name):
        super(StaticCase, self).__init__()
        paddle.enable_static()

    def runTest(self):
        pass

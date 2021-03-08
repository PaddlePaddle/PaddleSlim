import unittest
import paddle


class StaticCase(unittest.TestCase):
    def setUp(self):
        # switch mode
        paddle.enable_static()

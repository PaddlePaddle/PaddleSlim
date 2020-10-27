import unittest
import paddle


class StaticCase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

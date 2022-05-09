import os
import sys
sys.path.append("../")
import paddle
import unittest
from paddleslim.core import dygraph2program


class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D([1, 1])
        self.out = paddle.nn.Linear(256, 10)

    def forward(self, inputs):
        inputs = paddle.reshape(inputs, shape=[0, 1, 28, 28])
        y = self.conv(inputs)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, 256])
        y = self.out(y)
        return y


class TestEagerDygraph2Program(unittest.TestCase):
    def setUp(self):
        os.environ['FLAGS_enable_eager_mode'] = "1"
        self.prepare_inputs()
        self.prepare_layer()

    def prepare_inputs(self):
        self.inputs = [3, 28, 28]

    def prepare_layer(self):
        self.layer = Model()

    def test_dy2prog(self):
        program = dygraph2program(self.layer, self.inputs)
        self.assert_program(program)

    def assert_program(self, program):
        ops = [
            'reshape2', 'conv2d', 'elementwise_add', 'pool2d', 'reshape2',
            'matmul_v2', 'elementwise_add'
        ]
        self.assertListEqual([op.type for op in program.block(0).ops], ops)


class TestEagerDygraph2Program2(TestEagerDygraph2Program):
    def prepare_inputs(self):
        self.inputs = [[3, 28, 28]]


class TestEagerDygraph2Program3(TestEagerDygraph2Program):
    def prepare_inputs(self):
        self.inputs = paddle.randn([3, 28, 28])


class TestEagerDygraph2Program4(TestEagerDygraph2Program):
    def prepare_inputs(self):
        self.inputs = [paddle.randn([3, 28, 28])]


if __name__ == "__main__":
    unittest.main()

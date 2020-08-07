import sys
import paddle
import paddle.fluid as fluid
from paddleslim.quant import quant_aware, convert
import numpy as np

from paddle.fluid.layer_helper import LayerHelper


def pact(x, name=None):
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    init_thres = 20
    u_param_attr = fluid.ParamAttr(
        name=x.name + '_pact',
        initializer=fluid.initializer.ConstantInitializer(value=init_thres),
        regularizer=fluid.regularizer.L2Decay(0.0001),
        learning_rate=1)
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    x = fluid.layers.elementwise_sub(
        x, fluid.layers.relu(fluid.layers.elementwise_sub(x, u_param)))
    x = fluid.layers.elementwise_add(
        x, fluid.layers.relu(fluid.layers.elementwise_sub(-u_param, x)))

    return x


def get_optimizer():
    return fluid.optimizer.MomentumOptimizer(0.01, 0.9)

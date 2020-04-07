import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper


def dsq_round(x):
    #delta = np.max(x) - np.min(x)
    #x = (x/delta + 0.5)
    #return x.round() * 2 - 1
    return x


def dsq_round_back(dy):
    return np.array(dy)


def dsq(x, bit=8, name=None):
    def clip(x, upper, lower):
        x = x + fluid.layers.relu(lower - x)
        x = x - fluid.layers.relu(x - upper)
        return x

    def phi_function(x, mi, alpha, delta):
        s = 1 / (1 - alpha)
        k = fluid.layers.log(2 / alpha - 1) * (1 / delta)
        x = (fluid.layers.tanh((x - mi) * k)) * s
        return x

    def dequantize(x, lower_bound, delta, interval):

        # save mem
        x = ((x + 1) / 2 + interval) * delta + lower_bound
        return x

    helper = LayerHelper("dsq", **locals())
    dtype = 'float32'
    bit_range = 2**bit - 1

    u_param_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(value=10))
    l_param_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(value=-10))

    alpha_param_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(value=0.2))
    u_param = helper.create_parameter(
        attr=u_param_attr, shape=[1], dtype=dtype)
    l_param = helper.create_parameter(
        attr=l_param_attr, shape=[1], dtype=dtype)
    alpha_param = helper.create_parameter(
        attr=alpha_param_attr, shape=[1], dtype=dtype)

    upper = fluid.layers.create_global_var(
        shape=[1], value=10, dtype='float32', persistable=True)
    lower = fluid.layers.create_global_var(
        shape=[1], value=-10, dtype='float32', persistable=True)
    fluid.layers.assign(upper * 0.9 + u_param * 0.1, upper)
    fluid.layers.assign(lower * 0.9 + l_param * 0.1, lower)
    x = clip(x, upper, lower)
    delta = (upper - lower) / bit_range
    interval = (x - lower) // delta
    mi = (interval + 0.5) * delta + lower
    x = phi_function(x, mi, alpha_param, delta)
    out_var = fluid.default_main_program().current_block().create_var(
        name=x.name + '_dsq', dtype=dtype, shape=x.shape)
    fluid.layers.py_func(
        func=dsq_round,
        x=x,
        out=out_var,
        backward_func=dsq_round_back,
        skip_vars_in_backward_input=[x, out_var])
    x = dequantize(out_var, lower, delta, interval)

    return x


def dsq1(x, name=None):
    helper = LayerHelper("dsq1", **locals())
    dtype = 'float32'
    '''
    u_param_attr = fluid.ParamAttr(
            initializer=fluid.initializer.ConstantInitializer(value=10))
    u_param = helper.create_parameter(
        attr=u_param_attr,
        shape=[1],
        dtype=dtype)
    '''
    upper = fluid.layers.create_global_var(
        shape=[1], value=10, dtype='float32', persistable=True)
    x = x - fluid.layers.relu(x - upper)
    return x

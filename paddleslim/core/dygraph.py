import paddle
import numpy as np
from paddle.fluid.framework import _dygraph_tracer, dygraph_only, _dygraph_guard
from paddle.fluid.dygraph.base import program_desc_tracing_guard
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.framework import Block, ParamBase, Program, Variable

__all__ = ["dygraph2program"]


def _is_shape(values):
    if not isinstance(values, (list, tuple)):
        return False
    for v in values:
        if not isinstance(v, int):
            return False
    return True


def extract_vars(inputs):
    vars = []
    if isinstance(inputs, Variable):
        vars = [inputs]
    elif _is_shape(inputs):
        data = np.ones(tuple(inputs)).astype("float32")
        vars = [paddle.to_tensor(data)]
    elif isinstance(inputs, dict):
        vars = [paddle.to_tensor(value) for value in inputs.values()]
    return vars


@dygraph_only
def dygraph2program(layer,
                    inputs,
                    feed_prefix='feed_',
                    fetch_prefix='fetch_',
                    tmp_prefix='t_',
                    extract_vars_fn=None):
    assert isinstance(layer, Layer)

    extract_vars_fn = extract_vars_fn if extract_vars_fn is not None else extract_vars
    tracer = _dygraph_tracer()._get_program_desc_tracer()

    var_list = extract_vars_fn(inputs)

    with program_desc_tracing_guard(True):
        original_outputs = layer(*var_list)
        out_vars = extract_vars(original_outputs)
        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(
            var_list, feed_prefix, out_vars, fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = Program()
        program.desc = program_desc
        program.blocks = [Block(program, 0)]
        program._sync_with_cpp()
        return program

    return program

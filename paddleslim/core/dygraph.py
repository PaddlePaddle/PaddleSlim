import os
import paddle
import collections
import logging
import numpy as np
from paddle.fluid.framework import _dygraph_tracer, dygraph_only, _dygraph_guard, program_guard
from paddle.fluid.dygraph.base import program_desc_tracing_guard, _switch_declarative_mode_guard_
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.framework import Block, ParamBase, Program, Variable
from ..common import get_logger

__all__ = ["dygraph2program"]

_logger = get_logger(__name__, level=logging.INFO)


class NameGenerator:
    def __init__(self):
        self.ids = collections.defaultdict(int)

    def name(self, prefix):
        assert isinstance(prefix, str)

        name = "{}_{}".format(prefix, self.ids[prefix])
        self.ids[prefix] += 1

        return name


NG = NameGenerator()


def _is_shape(values):
    if not isinstance(values, (list, tuple)):
        return False
    for v in values:
        if not isinstance(v, int):
            return False
    return True


def _is_shapes(values):
    if not isinstance(values, (list, tuple)):
        return False
    for v in values:
        if not _is_shape(v):
            return False
    return True


def _create_tensors(shapes, dtypes=None, is_static=False):
    if dtypes is not None:
        assert len(shapes) == len(
            dtypes
        ), "Length of shapes and dtypes must be same. But get len(shapes): {}; len(dtypes): {}; shapes: {}; dtypes: {}".format(
            len(shapes), len(dtypes), shapes, dtypes)
    else:
        dtypes = len(shapes) * ['float32']
    tensors = []
    for shape, dtype in zip(shapes, dtypes):
        if is_static:
            tensors.append(
                paddle.static.data(
                    shape=shape, dtype=dtype, name=NG.name("feed")))
        else:
            data = np.ones(tuple(shape)).astype(dtype)
            tensors.append(paddle.to_tensor(data))
    return tensors


def extract_vars(inputs):
    """
    Extract a list of variables from inputs.
    Args:
        inputs(Variable | list<Object> | dict): 
    """
    vars = []
    if isinstance(inputs, Variable):
        vars = [inputs]
    elif isinstance(inputs, dict):
        for _key, _value in inputs.items():
            if isinstance(_value, Variable):
                vars.append(_value)
            else:
                _logger.warn(
                    f"Variable is excepted, but get an element with type({type(_value)}) from inputs whose type is dict. And the key of element is {_key}."
                )
    elif isinstance(inputs, (tuple, list)):

        for _value in inputs:
            vars.extend(extract_vars(_value))
    if len(vars) == 0:
        _logger.warn(f"Extract none variables from inputs.")
    return vars


def _to_var(x):
    """
    Convert Variable or np.array into Placeholder.
    """
    shape = x.shape
    dtype = x.dtype
    name = getattr(x, "name", None) or NG.name("feed")
    return paddle.static.data(shape=shape, dtype=dtype, name=name)


def to_variables(inputs, is_static=False):
    """
    Find and rename variables. Find np.ndarray and convert it to variable.
    """
    if isinstance(inputs,
                  (Variable, paddle.Tensor)) or isinstance(inputs, np.ndarray):
        if is_static:
            return _to_var(inputs)
        else:
            return paddle.fluid.dygraph.to_variable(inputs)
    elif isinstance(inputs, dict):
        ret = {}
        for _key in inputs:
            ret[_key] = to_variables(inputs[_key], is_static)
        return inputs
    elif isinstance(inputs, list):
        ret = []
        for _value in inputs:
            ret.append(to_variables(_value, is_static))
        return ret


@dygraph_only
def dygraph2program(layer,
                    inputs,
                    feed_prefix='feed_',
                    fetch_prefix='fetch_',
                    tmp_prefix='t_',
                    extract_inputs_fn=None,
                    extract_outputs_fn=None,
                    dtypes=None):
    print(type(layer))
    assert isinstance(layer, Layer)
    extract_inputs_fn = extract_inputs_fn if extract_inputs_fn is not None else extract_vars
    extract_outputs_fn = extract_outputs_fn if extract_outputs_fn is not None else extract_vars

    if os.environ.get("FLAGS_enable_eager_mode") == "1":
        return _dy2prog(layer, inputs, feed_prefix, fetch_prefix, tmp_prefix,
                        extract_inputs_fn, extract_outputs_fn, dtypes)

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    with program_desc_tracing_guard(True):

        if _is_shape(inputs):
            shapes = [inputs]
            inputs = _create_tensors(shapes, dtypes=dtypes)
            input_var_list = inputs
        elif _is_shapes(inputs):
            inputs = _create_tensors(inputs, dtypes=dtypes)
            input_var_list = inputs
        else:
            inputs = to_variables(inputs)
            input_var_list = extract_inputs_fn(inputs)

        original_outputs = layer(*inputs)
        # 'original_outputs' may be dict, so we should convert it to list of varibles.
        # And should not create new varibles in 'extract_vars'.
        out_var_list = extract_outputs_fn(original_outputs)
        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(
            input_var_list, feed_prefix, out_var_list, fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = Program()
        program.desc = program_desc
        program.blocks = [Block(program, 0)]
        program._sync_with_cpp()
    return program


def _dy2prog(layer,
             inputs,
             feed_prefix='feed_',
             fetch_prefix='fetch_',
             tmp_prefix='t_',
             extract_inputs_fn=None,
             extract_outputs_fn=None,
             dtypes=None):
    """
    Tracing program in Eager Mode.
    """
    paddle.enable_static()

    program = Program()
    # convert ParamBase into Parameter automatically by _switch_declarative_mode_guard_
    with program_guard(program), _switch_declarative_mode_guard_(True):
        if _is_shape(inputs):
            shapes = [inputs]
            inputs = _create_tensors(shapes, dtypes=dtypes, is_static=True)
        elif _is_shapes(inputs):
            inputs = _create_tensors(inputs, dtypes=dtypes, is_static=True)
        else:
            inputs = to_variables(inputs, is_static=True)
            inputs = extract_inputs_fn(inputs)
        outputs = layer(*inputs)

    paddle.disable_static()

    return program

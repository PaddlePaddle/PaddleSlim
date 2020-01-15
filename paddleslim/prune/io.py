import os
import paddle.fluid as fluid
from paddle.fluid import Program
from ..core import GraphWrapper
from ..common import get_logger
import json
import logging

__all__ = ["save_model", "load_model"]

_logger = get_logger(__name__, level=logging.INFO)


def save_model(graph, dirname):
    """
    Save weights of model and information of shapes into filesystem.

    Args:
      - graph(Program|Graph): The graph to be saved.
      - dirname(str): The directory that the model saved into.
    """
    assert graph is not None and dirname is not None
    graph = GraphWrapper(graph) if isinstance(graph, Program) else graph
    exe = fluid.Executor(fluid.CPUPlace())
    params_file = "__params__"
    shapes_file = "__shapes__"
    fluid.io.save_params(
        executor=exe,
        dirname=dirname,
        main_program=graph.program,
        filename=params_file)
    weights_file = os.path.join(dirname, params_file)
    _logger.info("Save model weights into {}".format(weights_file))
    shapes = {}
    for var in graph.all_parameters():
        shapes[var.name()] = var.shape()
    shapes_file = os.path.join(dirname, shapes_file)
    with open(shapes_file, "w") as f:
        json.dump(shapes, f)
        _logger.info("Save shapes of weights into {}".format(shapes_file))


def load_model(graph, dirname):
    """
    Load weights of model and information of shapes from filesystem.

    Args:
      - graph(Program|Graph): The graph to be saved.
      - dirname(str): The directory that the model saved into.
    """
    assert graph is not None and dirname is not None
    graph = GraphWrapper(graph) if isinstance(graph, Program) else graph
    exe = fluid.Executor(fluid.CPUPlace())
    params_file = "__params__"
    shapes_file = "__shapes__"

    shapes_file = os.path.join(dirname, shapes_file)
    _logger.info("Load shapes of weights from {}".format(shapes_file))
    with open(shapes_file, "r") as f:
        shapes = json.load(f)
        for param, shape in shapes.items():
            graph.var(param).set_shape(shape)

    _logger.info("Load shapes of weights from {}".format(shapes_file))

    exe = fluid.Executor(fluid.CPUPlace())
    fluid.io.load_params(
        executor=exe,
        dirname=dirname,
        main_program=graph.program,
        filename=params_file)
    graph.update_groups_of_conv()
    graph.infer_shape()
    _logger.info("Load weights from {}".format(
        os.path.join(dirname, params_file)))

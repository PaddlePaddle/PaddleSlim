import numpy as np
import logging
import paddle
from paddle.fluid.dygraph import TracedLayer
from ..core import GraphWrapper
from ..prune import collect_convs
from ..common import get_logger

__all__ = ["VarGroup"]

_logger = get_logger(__name__, level=logging.INFO)


class VarGroup():
    def __init__(self, model, input_shape):
        self.groups = []
        self._parse_model(model, input_shape)

    def find_group(self, var_name, axis):
        for group in self.groups:
            for _name, _axis, _ in group:
                if isinstance(_axis, int):
                    _axis = [_axis]  # TODO: fix
                print("_name: {}; var_name: {}; _axis: {}; axis: {}".format(
                    _name, var_name, _axis, axis))
                if _name == var_name and _axis == axis:
                    return group

    def _parse_model(self, model, input_shape):
        _logger.info("Parsing model with input: {}".format(input_shape))
        data = np.ones(tuple(input_shape)).astype("float32")
        in_var = paddle.to_tensor(data)
        out_dygraph, static_layer = TracedLayer.trace(model, inputs=[in_var])
        graph = GraphWrapper(static_layer.program)

        visited = {}
        for name, param in model.named_parameters():
            group = collect_convs([param.name], graph,
                                  visited)[0]  # [(name, axis, pruned_idx)]
            if len(group) > 0:
                self.groups.append(group)
        _logger.info("Found {} groups.".format(len(self.groups)))

    def __str__(self):
        return "\n".join([str(group) for group in self.groups])

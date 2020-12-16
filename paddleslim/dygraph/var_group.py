import numpy as np
import logging
import paddle
from paddle.fluid.dygraph import TracedLayer
from ..core import GraphWrapper, dygraph2program
from ..prune import collect_convs
from ..common import get_logger

__all__ = ["VarGroup"]

_logger = get_logger(__name__, level=logging.INFO)


class VarGroup():
    """
    A tool used to parse dygraph and store information of variables' relationship.
    Args:
      - model(nn.Layer): The dygraph to be parsed.
      - inputs(Variable|list|dict): The dummy inputs of target model. It will be used in calling `model.forward(inputs)`.
    """

    def __init__(self, model, inputs, extract_vars_fn=None):
        self.groups = []
        self._parse_model(model, inputs, extract_vars_fn=extract_vars_fn)

    def _to_dict(self, group):
        ret = {}
        for _name, _axis, _transforms in group:
            if isinstance(_axis, int):
                _axis = [_axis]  # TODO: fix
            ret[_name] = {'pruned_dims': _axis, 'transforms': _transforms}
        return ret

    def find_group(self, var_name, axis):
        for group in self.groups:
            for _name, _axis, _stride in group:
                if isinstance(_axis, int):
                    _axis = [_axis]  # TODO: fix
                if _name == var_name and _axis == axis:
                    return self._to_dict(group)

    def _parse_model(self, model, inputs, extract_vars_fn=None):
        _logger.debug("Parsing model with input: {}".format(inputs))

        model.eval()
        program = dygraph2program(
            model, inputs=inputs, extract_vars_fn=extract_vars_fn)
        graph = GraphWrapper(program)

        visited = {}
        for name, param in model.named_parameters():
            group = collect_convs([param.name], graph,
                                  visited)[0]  # [(name, axis, pruned_idx)]
            if len(group) > 0:
                self.groups.append(group)
        _logger.info("Found {} groups.".format(len(self.groups)))

    def __str__(self):
        return "\n".join([str(group) for group in self.groups])

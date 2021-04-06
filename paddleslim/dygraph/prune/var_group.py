import numpy as np
import logging
import paddle
from paddle.fluid.dygraph import TracedLayer
from paddleslim.core import GraphWrapper, dygraph2program
from paddleslim.prune import collect_convs
from paddleslim.common import get_logger

__all__ = ["VarGroup"]

_logger = get_logger(__name__, level=logging.INFO)


class VarGroup():
    """
    A tool used to parse dygraph and store information of variables' relationship.
    Args:
      - model(nn.Layer): The dygraph to be parsed.
      - inputs(Variable|list|dict): The dummy inputs of target model. It will be used in calling `model.forward(inputs)`.
    """

    def __init__(self, model, inputs):
        self.groups = []
        self._parse_model(model, inputs)

    def _to_dict(self, group):
        ret = {}
        for _name, _axis, _transforms, _op in group:
            if isinstance(_axis, int):
                _axis = [_axis]
            if _name not in ret:
                ret[_name] = []
            # Variable can be pruned on multiple axies.
            ret[_name].append({
                'pruned_dims': _axis,
                'transforms': _transforms,
                'op': _op
            })
        return ret

    def find_group(self, var_name, axis):
        for group in self.groups:
            prune_infos = group.get_prune_info(var_name, axis)
            if prune_infos is not None:
                return self._to_dict([(prune_info.name, prune_info.axis,
                                       prune_info.transform, prune_info.op)
                                      for prune_info in group.all_prune_info()])

    def _parse_model(self, model, inputs):
        _logger.debug("Parsing model with input: {}".format(inputs))
        # model can be in training mode, because some model contains auxiliary parameters for training.
        program = dygraph2program(model, inputs=inputs)
        graph = GraphWrapper(program)
        visited = {}
        params = [_param.name for _param in model.parameters()]
        self.groups = collect_convs(params, graph)
        _logger.info("Found {} groups.".format(len(self.groups)))

    def __str__(self):
        return "\n".join([str(group) for group in self.groups])

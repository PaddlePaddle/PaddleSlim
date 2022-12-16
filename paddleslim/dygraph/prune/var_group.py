import numpy as np
import logging
import paddle
from paddleslim.core import GraphWrapper, dygraph2program
from paddleslim.prune import PruningCollections
from paddleslim.common import get_logger

__all__ = ["DygraphPruningCollections"]

_logger = get_logger(__name__, level=logging.INFO)


class DygraphPruningCollections(PruningCollections):
    """
    A tool used to parse dygraph and store information of variables' relationship.
    Args:
      - model(nn.Layer): The dygraph to be parsed.
      - inputs(Variable|list|dict): The dummy inputs of target model. It will be used in calling `model.forward(inputs)`.
      - skip_leaves(bool): Whether to skip the last convolution layers.
    """

    def __init__(self,
                 model,
                 inputs,
                 skip_leaves=True,
                 prune_type='conv',
                 input_dtype='float32'):
        assert prune_type in ['conv', 'fc'
                              ], "Please select conv or fc as your prune type."
        _logger.debug("Parsing model with input: {}".format(inputs))
        # model can be in training mode, because some model contains auxiliary parameters for training.
        #TODO(minghaoBD): support dictionary input
        if isinstance(inputs[0], int):
            dtypes = [input_dtype]
        elif isinstance(inputs[0], list):
            dtypes = [input_dtype] * len(inputs)
        else:
            dtypes = [input_dtype]
        program = dygraph2program(model, inputs=inputs, dtypes=dtypes)

        graph = GraphWrapper(program)
        if prune_type == 'conv':
            params = [
                _param.name for _param in model.parameters()
                if len(_param.shape) == 4
            ]
        elif prune_type == 'fc':
            params = [
                _param.name for _param in model.parameters()
                if len(_param.shape) == 2
            ]
        self._collections = self.create_pruning_collections(
            params, graph, skip_leaves=skip_leaves, prune_type=prune_type)
        _logger.info("Found {} collections.".format(len(self._collections)))

        _name2values = {}
        for param in model.parameters():
            _name2values[param.name] = np.array(param.value().get_tensor())
        for collection in self._collections:
            collection.values = _name2values

    def find_collection_by_master(self, var_name, axis):
        for _collection in self._collections:
            if _collection.master['name'] == var_name and _collection.master[
                    'axis'] == axis:
                return _collection

    def __str__(self):
        return "\n".join(
            [str(_collection) for _collection in self._collections])

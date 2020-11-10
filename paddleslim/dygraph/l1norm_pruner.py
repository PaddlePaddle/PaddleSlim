import logging
import numpy as np
from ..common import get_logger
from .var_group import *
from .pruning_plan import *
from .pruner import Pruner

__all__ = ['L1NormPruner']

_logger = get_logger(__name__, logging.INFO)


class L1NormPruner(Pruner):
    def __init__(self, model, input_shape, sensitive=False):
        super(L1NormPruner, self).__init__(model, input_shape)
        self.model = model
        self.var_group = VarGroup(model, input_shape)
        self.sensitive = sensitive

    def prune_var(self, var_name, axis, pruned_ratio):
        if isinstance(axis, int):
            axis = [axis]
        group = self.var_group.find_group(var_name, axis)
        _logger.info("found group with {}: {}".format(var_name, group))
        plan = PruningPlan(self.model.full_name)
        for param in self.model.parameters():
            if var_name == param.name:
                value = np.array(param.value().get_tensor())
                reduce_axis = [
                    i for i in range(len(value.shape)) if i not in axis
                ]
                print("reduce_axis: {}".format(reduce_axis))
                l1norm = np.mean(np.abs(value), axis=tuple(reduce_axis))

                sorted_idx = l1norm.argsort()

                pruned_num = int(round(len(sorted_idx) * pruned_ratio))
                pruned_idx = sorted_idx[:pruned_num]
                mask_shape = [value.shape[i] for i in axis]
                print("mask shape: {}".format(mask_shape))
                mask = np.ones(mask_shape, dtype="int32")
                mask[pruned_idx] = 0

                plan.add(var_name, PruningMask(axis, mask))
        # TODO: add other variables of current group into plan.
        for var, dims, _ in group:
            if isinstance(dims, int):
                dims = [dims]
            plan.add(var, PruningMask(dims, mask))

        return plan

    def prune():
        pass

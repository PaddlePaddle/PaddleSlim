import logging
import numpy as np
import paddle
from ..common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner

__all__ = ['L2NormFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class L2NormFilterPruner(FilterPruner):
    def __init__(self, model, inputs, extract_vars_fn=None, sen_file=None):
        super(L2NormFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file, extract_vars_fn=extract_vars_fn)

    def cal_mask(self, var_name, pruned_ratio, group):
        value = group[var_name]['value']
        pruned_dims = group[var_name]['pruned_dims']
        reduce_dims = [
            i for i in range(len(value.shape)) if i not in pruned_dims
        ]

        # scores = np.mean(np.abs(value), axis=tuple(reduce_dims))
        scores = np.sqrt(np.sum(np.square(value), axis=tuple(reduce_dims)))
        sorted_idx = scores.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]
        mask_shape = [value.shape[i] for i in pruned_dims]
        mask = np.ones(mask_shape, dtype="int32")
        mask[pruned_idx] = 0
        return mask

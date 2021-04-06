import logging
import numpy as np
import paddle
from paddleslim.common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner

__all__ = ['L1NormFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class L1NormFilterPruner(FilterPruner):
    def __init__(self, model, inputs, sen_file=None):
        super(L1NormFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file)

    def cal_mask(self, var_name, pruned_ratio, group):
        for _item in group[var_name]:
            if _item['pruned_dims'] == [0]:
                value = _item['value']
                pruned_dims = _item['pruned_dims']

        groups = 1
        for _name in group:
            for _item in group[_name]:
                if _item['pruned_dims'] == [1] and "op" in _item:
                    groups = _item['op'].attr('groups')
                    if groups is not None and groups > 1:
                        break

        reduce_dims = [
            i for i in range(len(value.shape)) if i not in pruned_dims
        ]
        l1norm = np.mean(np.abs(value), axis=tuple(reduce_dims))
        if groups > 1:
            l1norm = l1norm.reshape([groups, -1])
            l1norm = np.mean(l1norm, axis=1)

        sorted_idx = l1norm.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]

        mask_shape = [value.shape[i] for i in pruned_dims]
        mask = np.ones(mask_shape, dtype="int32")
        if groups > 1:
            mask = mask.reshape([groups, -1])
        mask[pruned_idx] = 0
        return mask.reshape(mask_shape)

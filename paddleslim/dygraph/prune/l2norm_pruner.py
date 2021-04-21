import logging
import numpy as np
import paddle
from paddleslim.common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner

__all__ = ['L2NormFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class L2NormFilterPruner(FilterPruner):
    def __init__(self, model, inputs, sen_file=None):
        super(L2NormFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file)

    def cal_mask(self, var_name, pruned_axis, pruned_ratio, group):
        value = group[var_name][pruned_axis]['value']
        groups = 1
        for _name in group:
            for _item in group[_name]:
                assert (isinstance(_item['pruned_dims'], int))
                if _item['pruned_dims'] == 1 and "op" in _item:
                    groups = _item['op'].attr('groups')
                    if groups is not None and groups > 1:
                        break

        reduce_dims = [i for i in range(len(value.shape)) if i != pruned_axis]
        #l1norm = np.mean(np.abs(value), axis=tuple(reduce_dims))
        scores = np.sqrt(np.sum(np.square(value), axis=tuple(reduce_dims)))
        if groups > 1:
            scores = scores.reshape([groups, -1])
            scores = np.mean(scores, axis=1)

        sorted_idx = scores.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]

        mask_shape = [value.shape[pruned_axis]]
        mask = np.ones(mask_shape, dtype="int32")
        if groups > 1:
            mask = mask.reshape([groups, -1])
        mask[pruned_idx] = 0
        return mask.reshape(mask_shape)

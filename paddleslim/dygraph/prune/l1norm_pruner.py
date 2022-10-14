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
    def __init__(self, model, inputs, sen_file=None, opt=None,
                 skip_leaves=True):
        super(L1NormFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file, opt=opt, skip_leaves=skip_leaves)

    def cal_mask(self, pruned_ratio, collection, reshape=-1):
        var_name = collection.master_name
        pruned_axis = collection.master_axis
        value = collection.values[var_name]

        if value.shape[1] == 3 * value.shape[0]:
            reshape = reshape
        else:
            reshape = -1
        groups = 1
        for _detail in collection.all_pruning_details():
            assert (isinstance(_detail.axis, int))
            if _detail.axis == 1:
                _groups = _detail.op.attr('groups')
                if _groups is not None and _groups > 1:
                    groups = _groups
                    break

        original_shape = value.shape
        if reshape > 0:
            k = int(value.size / value.shape[0] / reshape)
            value = value.reshape((value.shape[0], reshape, k))

        reduce_dims = [i for i in range(len(value.shape)) if i != pruned_axis]
        l1norm = np.mean(np.abs(value), axis=tuple(reduce_dims))
        if groups > 1:
            l1norm = l1norm.reshape([groups, -1])
            l1norm = np.mean(l1norm, axis=1)

        sorted_idx = l1norm.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_num = min(len(sorted_idx) - 1, pruned_num)
        pruned_idx = sorted_idx[:pruned_num]

        mask_shape = [value.shape[pruned_axis]]
        mask = np.ones(mask_shape, dtype="int32")
        if groups > 1:
            mask = mask.reshape([groups, -1])
        mask[pruned_idx] = 0

        if reshape != -1:
            mask = np.repeat(mask[:, np.newaxis], k, axis=1)
            return mask.flatten()

        return mask.reshape(mask_shape)

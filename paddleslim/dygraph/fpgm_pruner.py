import logging
import numpy as np
import paddle
from ..common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner

__all__ = ['FPGMFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class FPGMFilterPruner(FilterPruner):
    def __init__(self, model, input_shape, extract_vars_fn=None, sen_file=None):
        super(FPGMFilterPruner, self).__init__(
            model,
            input_shape,
            extract_vars_fn=extract_vars_fn,
            sen_file=sen_file)

    def cal_mask(self, var_name, pruned_ratio, group):
        value = group[var_name]['value']
        pruned_dims = group[var_name]['pruned_dims']
        assert (pruned_dims == [0])

        dist_sum_list = []
        for out_i in range(value.shape[0]):
            dist_sum = self.get_distance_sum(value, out_i)
            dist_sum_list.append(dist_sum)
        scores = np.array(dist_sum_list)

        sorted_idx = scores.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]
        mask_shape = [value.shape[i] for i in pruned_dims]
        mask = np.ones(mask_shape, dtype="int32")
        mask[pruned_idx] = 0
        return mask

    def get_distance_sum(self, value, out_idx):
        w = value.view()
        w.shape = value.shape[0], np.product(value.shape[1:])
        selected_filter = np.tile(w[out_idx], (w.shape[0], 1))
        x = w - selected_filter
        x = np.sqrt(np.sum(x * x, -1))
        return x.sum()

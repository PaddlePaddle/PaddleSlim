import numpy as np
import logging
from .var_group import VarGroup
from ..common import get_logger

_logger = get_logger(__name__, level=logging.INFO)


class Status():
    def __init__(self):
        self.sensitivies = {}
        self.accumulates = None


class Pruner(object):
    def __init__(self, model, input_shape, sensitive=False, status=None):
        self.model = model
        self.input_shape = input_shape
        self.var_group = VarGroup(model, input_shape)
        self._status = status
        self.sensitive = sensitive

    def status(self, data=None, eval_func=None):
        if self._status is not None:
            return self._status
        self._status = Status()

        if self.sensitive:
            self.cal_sensitive(self.model, eval_func)
        return self._status

    def cal_sensitive(self, model, eval_func):
        sensitivities = self._status.sensitivies
        baseline = eval_func()
        ratios = np.arange(0.1, 1, step=0.1)
        for group in self.var_group.groups:
            var_name = group[0][0]
            dims = group[0][1]
            if var_name not in sensitivities:
                sensitivities[var_name] = {}
            for ratio in ratios:
                if ratio in sensitivities[var_name]:
                    _logger.debug("{}, {} has computed.".format(var_name,
                                                                ratio))
                    continue
                plan = self.prune_var(var_name, dims, ratio)
                plan.apply(model, lazy=True)
                pruned_metric = eval_func()
                loss = (baseline - pruned_metric) / baseline
                _logger.info("pruned param: {}; {}; loss={}".format(
                    var_name, ratio, loss))
                sensitivities[var_name][ratio] = loss
                plan.restore(model)
        return sensitivities

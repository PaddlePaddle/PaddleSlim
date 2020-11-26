import os
import pickle
import numpy as np
import logging
from .pruning_plan import PruningPlan
from ..common import get_logger

__all__ = ["Pruner"]

_logger = get_logger(__name__, level=logging.INFO)


class Pruner(object):
    """
    Pruner used to resize or mask dimensions of variables.
    Args:
        model(paddle.nn.Layer): The target model to be pruned.
        input_shape(list<int>): The input shape of model. It is used to trace the graph of the model.
        
    """

    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        self._var_shapes = {}
        for var in model.parameters():
            self._var_shapes[var.name] = var.shape
        self.plan = None

    def status(self, data=None, eval_func=None, status_file=None):
        raise NotImplemented("status is not implemented")

    def prune_var(self, var_name, axis, pruned_ratio):
        raise NotImplemented("prune_var is not implemented")

    def prune_vars(self, ratios, axis):
        """
        Pruning variables by given ratios.
        Args:
            ratios(dict<str, float>): The key is the name of variable to be pruned and the
                                      value is the pruned ratio.
            axis(list): The dimensions to be pruned on.

        Returns:
            plan(PruningPlan): The pruning plan.
        """
        global_plan = PruningPlan(self.model.full_name)
        for var, ratio in ratios.items():
            if not global_plan.contains(var, axis):
                plan = self.prune_var(var, axis, ratio)
                global_plan.extend(plan)
        global_plan.apply(self.model)
        return global_plan

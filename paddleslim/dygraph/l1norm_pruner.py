import logging
import numpy as np
import paddle
from ..common import get_logger
from .var_group import *
from .pruning_plan import *
from .pruner import Pruner

__all__ = ['L1NormFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class L1NormPruner(Pruner):
    """
    Pruner used to prune structure according to l1-norm score of each structure.
    Parameters:
        model(dygraph): The target model to be pruned.
        input_shape(list<int>): The input shape of model. It is used to trace the graph of the model.
        sensitive(bool): Whether to prune each layer by different ratio that is
                         in direct proportion to sensitivity of each layer. 'False' means pruning
                         each layer by uniform ratio. Default: "False".
        status(Status): The status of model. It contains sensitivity information used
                        in sensitive pruning. It can be instance loaded from local file system.
                        If it is None, you should call 'status' function to get status instance
                        before call 'prune' function and 'prune_var' function. Default: None.
    
    """

    def __init__(self, model, input_shape, sensitive=False, status=None):
        super(L1NormPruner, self).__init__(
            model, input_shape, sensitive=sensitive, status=status)
        self.model = model
        self.var_group = VarGroup(model, input_shape)

    def prune_var(self, var_name, axis, pruned_ratio):
        """
        Pruning a variable.
        Parameters:
            var_name(str): The name of variable.
            axis(list<int>): The axies to be pruned. For convolution with format [out_c, in_c, k, k],
                             'axis=[0]' means pruning filters and 'axis=[0, 1]' means pruning kernels.
            pruned_ratio(float): The ratio of pruned values in one variable.

        Returns:
            plan: An instance of PruningPlan that can be applied on model by calling 'plan.apply(model)'.

        """

        if isinstance(axis, int):
            axis = [axis]
        group = self.var_group.find_group(var_name, axis)
        _logger.debug("found group with {}: {}".format(var_name, group))
        plan = PruningPlan(self.model.full_name)
        for param in self.model.parameters():
            if var_name == param.name:
                value = np.array(param.value().get_tensor())
                reduce_axis = [
                    i for i in range(len(value.shape)) if i not in axis
                ]
                l1norm = np.mean(np.abs(value), axis=tuple(reduce_axis))
                # TODO: flatten l1norm to support pruning multi-dims
                sorted_idx = l1norm.argsort()

                pruned_num = int(round(len(sorted_idx) * pruned_ratio))
                pruned_idx = sorted_idx[:pruned_num]
                mask_shape = [value.shape[i] for i in axis]
                mask = np.ones(mask_shape, dtype="int32")
                mask[pruned_idx] = 0

                plan.add(var_name, PruningMask(axis, mask, pruned_ratio))
        for var, dims, _ in group:
            if isinstance(dims, int):
                dims = [dims]
            plan.add(var, PruningMask(dims, mask, pruned_ratio))

        return plan


class L1NormFilterPruner(L1NormPruner):
    def __init__(self, model, input_shape, sensitive=False, status=None):
        super(L1NormFilterPruner, self).__init__(
            model, input_shape, sensitive=sensitive, status=status)
        self.pruned_axis = [0]
        self.layer_type = paddle.fluid.dygraph.nn.Conv2D
        self.var_name = "weight"

    def uniform_prune(self,
                      skip_vars=None,
                      ratio=None,
                      constrain_value=None,
                      constrain_func=None):
        global_plan = PruningPlan(self.model.full_name)
        if ratio is not None:
            for name, sub_layer in self.model.named_sublayers():
                if isinstance(sub_layer, self.layer_type):
                    for name, param in sub_layer.named_parameters(
                            include_sublayers=False):
                        if name == self.var_name and param.name not in skip_vars and not global_plan.contains(
                                param.name, self.pruned_axis):
                            plan = self.prune_var(param.name, self.pruned_axis,
                                                  ratio)
                            skip = False
                            for var in skip_vars:
                                if plan.contains(var, self.pruned_axis):
                                    skip = True
                            if not skip:
                                global_plan.extend(plan)
        return global_plan

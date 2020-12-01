import os
import logging
import numpy as np
import pickle
import paddle
from ..common import get_logger
from .var_group import *
from .pruning_plan import *
from .pruner import Pruner
from ..analysis import dygraph_flops as flops
from .var_group import VarGroup

__all__ = ['Status', 'FilterPruner']

_logger = get_logger(__name__, logging.INFO)

CONV_OP_TYPE = paddle.nn.Conv2D
FILTER_DIM = [0]
CONV_WEIGHT_NAME = "weight"


class Status():
    def __init__(self, src=None):
        self.sensitivies = {}
        self.accumulates = {}
        self.is_ckp = True
        if src is not None:
            self.load(src)

    def save(self, dst):
        with open(dst, 'wb') as f:
            pickle.dump(self, f)
            _logger.info("Save status into {}".format(dst))

    def load(self, src):
        with open(src, 'rb') as f:
            data = pickle.load(f)
            self.sensitivies = data.sensitivies
            self.accumulates = data.accumulates
            self.is_ckp = data.is_ckp
            _logger.info("Load status from {}".format(src))


class FilterPruner(Pruner):
    """
    Pruner used to prune filter structure in convolution layer.

    Args:
        model(paddle.nn.Layer): The target model to be pruned.
        input_shape(list<int>): The input shape of model. It is used to trace the graph of the model.
        sen_file(str, optional): The absolute path of file that stores computed sensitivities. If it is
                              set rightly, 'FilterPruner::sensitive' function can not be called anymore
                              in next step. Default: None.
    
    """

    def __init__(self, model, input_shape, sen_file=None):
        super(FilterPruner, self).__init__(model, input_shape)
        self._status = Status(sen_file)
        # sensitive and var_group are just used in filter pruning
        self.var_group = VarGroup(model, input_shape)

    def sensitive(self,
                  eval_func=None,
                  sen_file=None,
                  target_vars=None,
                  skip_vars=None):
        """
        Compute or get sensitivities of model in current pruner. It will return a cached sensitivities when all the arguments are "None".

        This function return a dict storing sensitivities as below:
    
        .. code-block:: python
    
               {"weight_0":
                   {0.1: 0.22,
                    0.2: 0.33
                   },
                 "weight_1":
                   {0.1: 0.21,
                    0.2: 0.4
                   }
               }
    
        ``weight_0`` is parameter name of convolution. ``sensitivities['weight_0']`` is a dict in which key is pruned ratio and value is the percent of losses.

        Args:
          eval_func(function, optional): The function to evaluate the model in current pruner. This function should have an empy arguments list and return a score with type "float32". Default: None.
          sen_file(str, optional): The absolute path of file to save sensitivities into local filesystem. Default: None.
          target_vars(list, optional): The names of tensors whose sensitivity will be computed. "None" means all weights in convolution layer will be computed. Default: None.
          skip_vars(list, optional): The names of tensors whose sensitivity won't be computed. "None" means skip nothing. Default: None.
    
        Returns:
           dict: A dict storing sensitivities.       

        """
        if eval_func is None and sen_file is None:
            return self._status.sensitivies
        if sen_file is not None and os.path.isfile(sen_file):
            self._status.load(sen_file)

        if not self._status.is_ckp:
            return self._status

        self._cal_sensitive(
            self.model,
            eval_func,
            status_file=sen_file,
            target_vars=target_vars,
            skip_vars=skip_vars)

        self._status.is_ckp = False
        return self._status.sensitivies

    def _get_ratios_by_loss(self, sensitivities, loss, skip_vars=[]):
        """
        Get the max ratio of each parameter. The loss of accuracy must be less than given `loss`
        when the single parameter was pruned by the max ratio. 
        
        Args:
          
          sensitivities(dict): The sensitivities used to generate a group of pruning ratios. The key of dict
                               is name of parameters to be pruned. The value of dict is a list of tuple with
                               format `(pruned_ratio, accuracy_loss)`.
          loss(float): The threshold of accuracy loss.
          skip_vars(list, optional): The names of tensors whose sensitivity won't be computed. "None" means skip nothing. Default: None.
    
        Returns:
    
          dict: A group of ratios. The key of dict is name of parameters while the value is the ratio to be pruned.
        """
        ratios = {}
        for param, losses in sensitivities.items():
            if param in skip_vars:
                continue
            losses = losses.items()
            losses = list(losses)
            losses.sort()
            for i in range(len(losses))[::-1]:
                if losses[i][1] <= loss:
                    if i == (len(losses) - 1):
                        ratios[param] = float(losses[i][0])
                    else:
                        r0, l0 = losses[i]
                        r1, l1 = losses[i + 1]
                        r0 = float(r0)
                        r1 = float(r1)
                        d0 = loss - l0
                        d1 = l1 - loss

                        ratio = r0 + (loss - l0) * (r1 - r0) / (l1 - l0)
                        ratios[param] = ratio
                        if ratio > 1:
                            _logger.info(losses, ratio, (r1 - r0) / (l1 - l0),
                                         i)

                    break
        return ratios

    def _round_to(self, ratios, dims=[0], factor=8):
        ret = {}
        for name in ratios:
            ratio = ratios[name]
            dim = self._var_shapes[name][dims[0]]
            remained = round((1 - ratio) * dim / factor) * factor
            if remained == 0:
                remained = factor
            ratio = float(dim - remained) / dim
            ratio = ratio if ratio > 0 else 0.
            ret[name] = ratio
        return ret

    def get_ratios_by_sensitivity(self,
                                  pruned_flops,
                                  align=None,
                                  dims=[0],
                                  skip_vars=[]):
        """
         Get a group of ratios by sensitivities.
         Args:
             pruned_flops(float): The excepted rate of FLOPs to be pruned. It should be in range (0, 1).
             align(int, optional): Round the size of each pruned dimension to multiple of 'align' if 'align' is not None. Default: None.
             dims(list, optional): The dims to be pruned on. [0] means pruning channels of output for convolution. Default: [0].
             skip_vars(list, optional): The names of tensors whose sensitivity won't be computed. "None" means skip nothing. Default: None.

        Returns:
            tuple: A tuple with format ``(ratios, pruned_flops)`` . "ratios" is a dict whose key is name of tensor and value is ratio to be pruned. "pruned_flops" is the ratio of total pruned FLOPs in the model.
        """
        base_flops = flops(self.model, self.input_shape)

        _logger.info("Base FLOPs: {}".format(base_flops))
        low = 0.
        up = 1.0
        history = set()
        while low < up:
            loss = (low + up) / 2
            ratios = self._get_ratios_by_loss(
                self._status.sensitivies, loss, skip_vars=skip_vars)
            _logger.debug("pruning ratios: {}".format(ratios))
            if align is not None:
                ratios = self._round_to(ratios, dims=dims, factor=align)
            plan = self.prune_vars(ratios, axis=dims)
            _logger.debug("pruning plan: {}".format(plan))
            c_flops = flops(self.model, self.input_shape)
            _logger.debug("FLOPs after pruning: {}".format(c_flops))
            c_pruned_flops = (base_flops - c_flops) / base_flops
            plan.restore(self.model)
            _logger.debug("Seaching ratios, pruned FLOPs: {}".format(
                c_pruned_flops))
            key = str(round(c_pruned_flops, 4))
            if key in history:
                return ratios, c_pruned_flops
            history.add(key)
            if c_pruned_flops < pruned_flops:
                low = loss
            elif c_pruned_flops > pruned_flops:
                up = loss
            else:
                return ratios, c_pruned_flops
        return ratios, c_pruned_flops

    def _cal_sensitive(self,
                       model,
                       eval_func,
                       status_file=None,
                       target_vars=None,
                       skip_vars=None):
        sensitivities = self._status.sensitivies
        baseline = eval_func()
        ratios = np.arange(0.1, 1, step=0.1)
        for group in self.var_group.groups:
            var_name = group[0][0]
            dims = group[0][1]

            if target_vars is not None and var_name not in target_vars:
                continue
            if skip_vars is not None and var_name in skip_vars:
                continue

            if var_name not in sensitivities:
                sensitivities[var_name] = {}
            for ratio in ratios:
                ratio = round(ratio, 2)
                if ratio in sensitivities[var_name]:
                    _logger.debug("{}, {} has computed.".format(var_name,
                                                                ratio))
                    continue
                plan = self.prune_var(var_name, dims, ratio, apply="lazy")
                pruned_metric = eval_func()
                loss = (baseline - pruned_metric) / baseline
                _logger.info("pruned param: {}; {}; loss={}".format(
                    var_name, ratio, loss))
                sensitivities[var_name][ratio] = loss
                self._status.save(status_file)
                plan.restore(model)

        return sensitivities

    def sensitive_prune(self, pruned_flops, skip_vars=[], align=None):

        # skip depthwise convolutions
        for layer in self.model.sublayers():
            if isinstance(layer,
                          paddle.nn.layer.conv.Conv2D) and layer._groups > 1:
                for param in layer.parameters(include_sublayers=False):
                    skip_vars.append(param.name)
        _logger.debug("skip vars: {}".format(skip_vars))
        self.restore()
        ratios, pruned_flops = self.get_ratios_by_sensitivity(
            pruned_flops, align=align, dims=FILTER_DIM, skip_vars=skip_vars)
        _logger.debug("ratios: {}".format(ratios))
        self.plan = self.prune_vars(ratios, FILTER_DIM)
        self.plan._pruned_flops = pruned_flops
        #        self.plan.apply(self.model)
        return self.plan

    def restore(self):
        if self.plan is not None:
            self.plan.restore(self.model)

    def cal_mask(self, var_name, pruned_ratio, group):
        """
        
        {
          var_name: {
                        'layer': sub_layer,
                        'var': variable,
                        'value': np.array([]),
                        'pruned_dims': [1],
                      }
        }
        """
        raise NotImplemented("cal_mask is not implemented")

    def prune_var(self, var_name, pruned_dims, pruned_ratio, apply="impretive"):
        """
        Pruning a variable.
        Parameters:
            var_name(str): The name of variable.
            pruned_dims(list<int>): The axies to be pruned. For convolution with format [out_c, in_c, k, k],
                             'axis=[0]' means pruning filters and 'axis=[0, 1]' means pruning kernels.
            pruned_ratio(float): The ratio of pruned values in one variable.

        Returns:
            plan: An instance of PruningPlan that can be applied on model by calling 'plan.apply(model)'.

        """

        if isinstance(pruned_dims, int):
            pruned_dims = [pruned_dims]
        group = self.var_group.find_group(var_name, pruned_dims)
        _logger.debug("found group with {}: {}".format(var_name, group))
        plan = PruningPlan(self.model.full_name)

        for sub_layer in self.model.sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                if param.name in group:
                    group[param.name]['layer'] = sub_layer
                    group[param.name]['var'] = param
                    group[param.name]['value'] = np.array(param.value()
                                                          .get_tensor())
                    _logger.debug(f"set value of {param.name} into group")

        mask = self.cal_mask(var_name, pruned_ratio, group)
        for _name in group:
            dims = group[_name]['pruned_dims']
            if isinstance(dims, int):
                dims = [dims]
            plan.add(_name, PruningMask(dims, mask, pruned_ratio))
        if apply == "lazy":
            plan.apply(self.model, lazy=True)
        elif apply == "impretive":
            plan.apply(self.model, lazy=False)
        return plan

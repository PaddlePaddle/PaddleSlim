import numpy as np
import logging
from ..common import get_logger
import paddle.fluid as fluid
_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['PruningPlan', 'PruningMask']


class PruningMask():
    def __init__(self, dims, mask):
        self.dims = dims
        self.mask = mask

    def __str__(self):
        return "dims:{}\nmask:{}".format(self.dims, self.mask)


class PruningPlan():
    def __init__(self, model_name=None):
        # {"conv_weight": (axies, mask)}

        self._model_name = model_name
        self._plan_id = model_name
        self._masks = {}  #{param_name: pruning_mask}
        self._pruned_size = None
        self._total_size = None
        self._pruned_flops = None
        self._pruned_size = None
        self._model_size = None

    def add(self, var_name, mask):
        assert (isinstance(mask, PruningMask))
        self._masks[var_name] = mask

    def __str__(self):
        return "\n".join([
            "name:{}\npruning plan:{}".format(name, mask)
            for name, mask in self._masks.items()
        ])

    def lazy_apply(self, model):
        for param in model.parameters():
            if param.name in self._masks:
                dims = self._masks[param.name].dims
                mask = self._masks[param.name].mask

                t_value = param.value().get_tensor()
                value = np.array(t_value).astype("float32")
                expand_mask_shape = [1] * len(value.shape)
                for i in dims:
                    expand_mask_shape[i] = value.shape[i]
                _logger.info("Expanded mask shape: {}".format(
                    expand_mask_shape))
                expand_mask = mask.reshape(expand_mask_shape).astype("float32")
                place = fluid.CUDAPlace(0)
                t_value.set(value * expand_mask, place)

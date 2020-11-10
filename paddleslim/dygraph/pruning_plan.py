import paddle
import numpy as np
import logging
from ..common import get_logger
import paddle.fluid as fluid
from paddle.fluid import core
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

    def apply(self, model, lazy=False):
        if lazy:
            self.lazy_apply(model)
        else:
            self.imperative_apply(model)

    def lazy_apply(self, model):
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                if param.name in self._masks:
                    dims = self._masks[param.name].dims
                    mask = self._masks[param.name].mask
                    t_value = param.value().get_tensor()
                    value = np.array(t_value).astype("float32")
                    # The name of buffer can not contains "."
                    sub_layer.register_buffer(
                        param.name.replace(".", "_") + "_backup",
                        paddle.to_tensor(value))
                    _logger.info("Backup values of {} into buffers.".format(
                        param.name))
                    expand_mask_shape = [1] * len(value.shape)
                    for i in dims:
                        expand_mask_shape[i] = value.shape[i]
                    _logger.info("Expanded mask shape: {}".format(
                        expand_mask_shape))
                    expand_mask = mask.reshape(expand_mask_shape).astype(
                        "float32")

                    p = t_value._place()
                    if p.is_cpu_place():
                        place = paddle.CPUPlace()
                    elif p.is_cuda_pinned_place():
                        place = paddle.CUDAPinnedPlace()
                    else:
                        p = core.Place()
                        p.set_place(t_value._place())
                        place = paddle.CUDAPlace(p.gpu_device_id())

                    t_value.set(value * expand_mask, place)

    def imperative_apply(self, model):
        """
        Pruning values of variable imperatively. It is valid when pruning
        on one dimension.
        """
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                if param.name in self._masks:
                    dims = self._masks[param.name].dims
                    mask = self._masks[param.name].mask
                    assert (
                        len(dims) == 1,
                        "Imperative mode only support for pruning"
                        "on one dimension, but get dims {} when pruning parameter {}".
                        format(dims, param.name))
                    t_value = param.value().get_tensor()
                    value = np.array(t_value).astype("float32")
                    # The name of buffer can not contains "."
                    sub_layer.register_buffer(
                        param.name.replace(".", "_") + "_backup",
                        paddle.to_tensor(value))
                    _logger.info("Backup values of {} into buffers.".format(
                        param.name))
                    bool_mask = mask.astype(bool)
                    pruned_value = np.apply_along_axis(
                        lambda data: data[~bool_mask], dims[0], value)
                    place = fluid.CUDAPlace(0)
                    t_value.set(pruned_value, place)

    def restore(self, model):
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                backup_name = "_".join([param.name.replace(".", "_"), "backup"])
                if backup_name in sub_layer._buffers:
                    _logger.info("Restore values of variable: {}".format(
                        param.name))
                    t_value = param.value().get_tensor()
                    t_backup = sub_layer._buffers[backup_name].value(
                    ).get_tensor()

                    p = t_value._place()
                    if p.is_cpu_place():
                        place = paddle.CPUPlace()
                    elif p.is_cuda_pinned_place():
                        place = paddle.CUDAPinnedPlace()
                    else:
                        p = core.Place()
                        p.set_place(t_value._place())
                        place = paddle.CUDAPlace(p.gpu_device_id())

                    t_value.set(np.array(t_backup).astype("float32"), place)
                    del sub_layer._buffers[backup_name]

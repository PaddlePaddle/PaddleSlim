import paddle
import collections
import numpy as np
import logging
from ..common import get_logger
from paddle.fluid import core
_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['PruningPlan', 'PruningMask']


class PruningMask():
    def __init__(self, dims, mask, ratio):
        self._dims = dims
        self._mask = mask
        self._pruned_ratio = ratio

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        if not isinstance(value, collections.Iterator):
            raise ValueError(
                "The dims of PruningMask must be instance of collections.Iterator."
            )
        if self._mask is not None:
            assert len(self._mask.shape) == len(
                value
            ), "The length of value must be same with shape of mask in current PruningMask instance."
        self._dims = list(value)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        assert (isinstance(value, PruningMask))
        if self._dims is not None:
            assert len(self._mask.shape) == len(
                value
            ), "The length of value must be same with shape of mask in current PruningMask instance."
        self._mask = value

    def __str__(self):
        return "{}\t{}".format(self._pruned_ratio, self._dims)


class PruningPlan():
    def __init__(self, model_name=None):
        # {"conv_weight": (axies, mask)}

        self._model_name = model_name
        self._plan_id = model_name
        self._masks = {}  #{param_name: pruning_mask}
        self._dims = {}
        self._pruned_size = None
        self._total_size = None
        self._pruned_flops = None
        self._pruned_size = None
        self._model_size = None

    @property
    def pruned_flops(self):
        return self._pruned_flops

    @pruned_flops.setter
    def pruned_flops(self, value):
        self._pruned_flops = value

    def add(self, var_name, pruning_mask):
        assert (isinstance(pruning_mask, PruningMask))
        if var_name not in self._masks:
            self._masks[var_name] = []
        self._masks[var_name].append(pruning_mask)
        if var_name not in self._dims:
            self._dims[var_name] = []
        self._dims[var_name].append(pruning_mask.dims)

    @property
    def masks(self):
        return self._masks

    def extend(self, plan):
        assert (isinstance(plan, PruningPlan))
        for var_name in plan.masks:
            for mask in plan.masks[var_name]:
                if not self.contains(var_name, mask.dims):
                    self.add(var_name, mask)

    def contains(self, var_name, dims=None):
        return (var_name in self._dims) and (dims is None or
                                             dims in self._dims[var_name])

    def __str__(self):
        details = "\npruned FLOPs: {}".format(self._pruned_flops)
        head = "variable name\tpruned ratio\tpruned dims\n"
        return head + "\n".join([
            "{}:\t{}".format(name, ",".join([str(m) for m in mask]))
            for name, mask in self._masks.items()
        ]) + details

    def apply(self, model, lazy=False):
        if lazy:
            self.lazy_apply(model)
        else:
            self.imperative_apply(model)

    def lazy_apply(self, model):
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                if param.name in self._masks:
                    for _mask in self._masks[param.name]:
                        dims = _mask.dims
                        mask = _mask.mask
                        t_value = param.value().get_tensor()
                        value = np.array(t_value).astype("float32")
                        # The name of buffer can not contains "."
                        backup_name = param.name.replace(".", "_") + "_backup"
                        if backup_name not in sub_layer._buffers:
                            sub_layer.register_buffer(backup_name,
                                                      paddle.to_tensor(value))
                            _logger.debug("Backup values of {} into buffers.".
                                          format(param.name))
                        expand_mask_shape = [1] * len(value.shape)
                        for i in dims:
                            expand_mask_shape[i] = value.shape[i]
                        _logger.debug("Expanded mask shape: {}".format(
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
                    for _mask in self._masks[param.name]:
                        dims = _mask.dims
                        mask = _mask.mask
                        assert len(
                            dims
                        ) == 1, "Imperative mode only support for pruning on one dimension, but get dims {} when pruning parameter {}".format(
                            dims, param.name)
                        t_value = param.value().get_tensor()
                        value = np.array(t_value).astype("float32")
                        # The name of buffer can not contains "."
                        backup_name = param.name.replace(".", "_") + "_backup"
                        if backup_name not in sub_layer._buffers:
                            sub_layer.register_buffer(backup_name,
                                                      paddle.to_tensor(value))
                            _logger.debug("Backup values of {} into buffers.".
                                          format(param.name))
                        bool_mask = mask.astype(bool)
                        pruned_value = np.apply_along_axis(
                            lambda data: data[bool_mask], dims[0], value)

                        p = t_value._place()
                        if p.is_cpu_place():
                            place = paddle.CPUPlace()
                        elif p.is_cuda_pinned_place():
                            place = paddle.CUDAPinnedPlace()
                        else:
                            p = core.Place()
                            p.set_place(t_value._place())
                            place = paddle.CUDAPlace(p.gpu_device_id())

                        t_value.set(pruned_value, place)
                        if isinstance(sub_layer, paddle.nn.layer.conv.Conv2D):
                            if sub_layer._groups > 1 and pruned_value.shape[
                                    1] == 1:  # depthwise conv2d
                                _logger.debug(
                                    "Update groups of depthwise conv2d form {} to {}".
                                    format(sub_layer._groups,
                                           pruned_value.shape[0]))
                                sub_layer._groups = pruned_value.shape[0]
                    # for training
                    if param.trainable:
                        param.clear_gradient()

    def restore(self, model):
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                backup_name = "_".join([param.name.replace(".", "_"), "backup"])
                if backup_name in sub_layer._buffers:
                    _logger.debug("Restore values of variable: {}".format(
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

                    if isinstance(sub_layer, paddle.nn.layer.conv.Conv2D):
                        if sub_layer._groups > 1:
                            _logger.debug(
                                "Update groups of conv form {} to {}".format(
                                    sub_layer._groups, t_value.shape()[0]))
                            sub_layer._groups = t_value.shape()[0]
                    del sub_layer._buffers[backup_name]

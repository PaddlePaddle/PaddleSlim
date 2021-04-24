import paddle
import collections
import numpy as np
import logging
from paddleslim.common import get_logger
from paddle.fluid import core
_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['PruningPlan', 'PruningMask']


class PruningMask():
    def __init__(self, dims, mask, ratio, op):
        assert (isinstance(dims, int))
        self._dims = dims
        self._mask = mask
        self._pruned_ratio = ratio
        self._op = op

    @property
    def dims(self):
        return self._dims

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
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
        if var_name not in self._dims:
            self._dims[var_name] = []

        if pruning_mask.dims in self._dims[var_name]:
            for _mask in self._masks[var_name]:
                if pruning_mask.dims == _mask.dims:
                    _mask.mask = list(
                        np.array(_mask.mask).astype(np.int64) & np.array(
                            pruning_mask.mask).astype(np.int64))
        else:
            self._masks[var_name].append(pruning_mask)
            self._dims[var_name].append(pruning_mask.dims)

    @property
    def masks(self):
        return self._masks

    def extend(self, plan):
        assert (isinstance(plan, PruningPlan))
        for var_name in plan.masks:
            for mask in plan.masks[var_name]:
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
                        expand_mask_shape[dims] = value.shape[dims]
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
                        assert (isinstance(dims, int))
                        mask = _mask.mask
                        bool_mask = np.array(mask).astype(bool)
                        t_value = param.value().get_tensor()
                        value = np.array(t_value).astype("float32")

                        groups = _mask._op.attr('groups')
                        if dims == 1 and groups is not None and groups > 1 and len(
                                value.shape) == 4:
                            filter_size = value.shape[1]
                            except_num = np.sum(bool_mask)
                            assert (except_num % filter_size == 0)
                            new_groups = int(except_num / filter_size)
                            sub_layer._origin_groups = sub_layer._groups
                            sub_layer._groups = new_groups
                            _logger.info("change groups from {} to {} for {}.".
                                         format(groups, new_groups, param.name))
                            continue

                        # The name of buffer can not contains "."
                        backup_name = param.name.replace(".", "_") + "_backup"
                        if backup_name not in sub_layer._buffers:
                            sub_layer.register_buffer(backup_name,
                                                      paddle.to_tensor(value))
                            _logger.debug("Backup values of {} into buffers.".
                                          format(param.name))
                        pruned_value = np.apply_along_axis(
                            lambda data: data[bool_mask], dims, value)
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
                    if "_origin_groups" in sub_layer.__dict__:
                        sub_layer._groups = sub_layer._origin_groups
                    del sub_layer._buffers[backup_name]

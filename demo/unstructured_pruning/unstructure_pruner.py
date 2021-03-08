import numpy as np
import paddle


class UnstructurePruner():
    """
    The unstructure pruner.
    Args:
      - program(paddle.static.Program): The model to be pruned.
      - scope(paddle.static.Scope): The scope storing values of all variables. None means paddle.static.global_scope. Default: None.
      - place(CPUPlace | CUDAPlace): The device place used to execute model. None means CPUPlace. Default: None.
    """

    def __init__(self, program, scope=None, place=None):
        self.scope = paddle.static.global_scope() if scope == None else scope
        self.place = paddle.static.CPUPlace() if place is None else place
        self.masks = self._apply_masks(program, self.mask_parameters)

    def mask_parameters(self, parameters, masks):
        """
        Update masks and parameters. It is executed before each iteration.
        User can overwrite this function in subclass to implememt different pruning stragies.

        Args:
          - parameters(list<Tensor>): The parameters to be pruned.
          - masks(list<Tensor>): The masks used to keep zero values in parameters.
        """
        for param, mask in zip(parameters, masks):
            paddle.assign(mask * (param != 0), output=mask)
            paddle.assign(param * mask, output=param)

    def _apply_masks(self, program, mask_func):
        params = []
        masks = []
        for param in program.all_parameters():

            mask = program.global_block().create_var(
                name=param.name + "_mask",
                shape=param.shape,
                dtype=param.dtype,
                type=param.type,
                persistable=param.persistable,
                stop_gradient=True)

            self.scope.var(param.name + "_mask").get_tensor().set(
                np.ones(mask.shape).astype("float32"), self.place)
            params.append(param)
            masks.append(mask)

        with paddle.static.program_guard(main_program=program):
            ops = paddle.static.default_main_program().global_block().ops
            ori_len = len(ops)
            mask_func(params, masks)
            ops = ops[:ori_len] + ops[ori_len:]
            program.global_block().ops = ops
        d_masks = {}
        for _param, _mask in zip(params, masks):
            d_masks[_param.name] = _mask.name
        return d_masks

    def prune_by_ratio(self, ratios):
        """
        Pruning layers by ratio.
        Args:
          - ratios(dict): The key is name of parameters and the value is the ratio to be pruned.
        """
        for param, ratio in ratios.items():
            assert param in self.masks, f"Can't found {param.name} in parameters of target program."
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            ori_shape = v_param.shape
            v_param = v_param.flatten()
            pruned_num = round(len(v_param) * ratio)
            v_param[np.abs(v_param).argsort()[:pruned_num]] = 0
            v_param = v_param.reshape(ori_shape)
            v_mask = (v_param != 0).astype(v_param.dtype)
            t_param.set(v_param, self.place)
            t_mask.set(v_mask, self.place)

    def uniform_prune(self, ratio):
        ratios = {}
        for param in self.masks:
            ratios[param] = ratio
        self.prune_by_ratio(ratios)

    def update_params(self):
        for param in self.masks:
            mask = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask).get_tensor()
            v_param = np.array(t_param) * np.array(t_mask)
            t_param.set(v_param, self.place)

    def sparse(self, program):
        """
        Get sparse ratio of each layer in program.
        Args:
          - program(paddle.static.Program): The target program to be pruned.
        Returns:
          - dict: The key is name of parameter and value is sparse ratio.
        """
        ratios = {}
        for param in program.all_parameters():
            mask = np.array(
                self.scope.find_var(param.name + "_mask").get_tensor())
            ratio = np.sum(mask) / float(np.product(mask.shape))
            ratios[param.name] = ratio
        return ratios

    def total_sparse(self, program):
        total = 0
        values = 0
        for param in program.all_parameters():
            mask = np.array(paddle.static.global_scope().find_var(
                param.name + "_mask").get_tensor())
            total += np.product(param.shape)
            values += np.sum(mask)
        return float(values) / total

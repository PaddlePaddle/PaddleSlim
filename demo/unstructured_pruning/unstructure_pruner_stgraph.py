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

    def __init__(self,
                 program,
                 mode,
                 threshold,
                 ratio_value=0.5,
                 threshold_value=1e-5,
                 scope=None,
                 place=None):
        self.threshold = threshold
        self.threshold_value = threshold_value
        self.mode = mode
        self.ratio_value = ratio_value
        assert self.mode in [
            'ratio', 'threshold'
        ], "mode must be selected from 'ratio' and 'threshold'"
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
            if not 'weights' in param.name: continue
            paddle.assign(
                mask * (paddle.abs(param) >= self.threshold), output=mask)
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

    def summarize_weights(self, program, ratio=0.1):
        """
        The function is used to get the weights corresponding to a given ratio
        when you are uncertain about the threshold in __init__() function above.
        For example, when given 0.1 as ratio, the function will print the weight value,
        the abs(weights) lower than which count for 10% of the total numbers.

        Args:
          - program(paddle.static.Program): The model which have all the parameters.
          - ratio(float): The ratio illustrated above.
        """
        data = []
        for param in program.all_parameters():
            data.append(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()).flatten())
        data = np.concatenate(data, axis=0)
        print(np.sort(np.abs(data))[int(ratio * len(data))])

    def sparse_by_layer(self, program):
        """
        The function is used to get the density at each layer, usually called for debuggings.
        
        Args:
          - program(paddle.static.Program): The current model.
        """
        layer_sparse = {}
        for param in program.all_parameters():
            mask = np.array(paddle.static.global_scope().find_var(
                param.name + '_mask').get_tensor())
            layer_sparse[param.name] = np.sum(mask) / np.product(param.shape)
        return layer_sparse

    def update_threshold(self):
        '''
        Update the threshold after each optimization step in RATIO mode.
        User should overwrite this method togther with self.mask_parameters()
        '''
        params_flatten = []
        for param in self.masks:
            if not 'weight' in param: continue
            t_param = self.scope.find_var(param).get_tensor()
            v_param = np.array(t_param)
            params_flatten.append(v_param.flatten())
        params_flatten = np.concatenate(params_flatten, axis=0)
        total_len = len(params_flatten)
        self.threshold_value = np.sort(np.abs(params_flatten))[max(
            0, int(self.ratio_value * total_len) - 1)]

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'threshold':
            return
        elif self.mode == 'ratio':
            self.update_threshold()

    def update_params(self):
        """
        Update the parameters given self.masks, usually called before saving models.
        """
        for param in self.masks:
            mask = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask).get_tensor()
            v_param = np.array(t_param) * np.array(t_mask)
            t_param.set(v_param, self.place)

    @staticmethod
    def total_sparse(program):
        """
        The function is used to get the whole model's density (1-sparsity).
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        Args:
          - program(paddle.static.Program): The current model.
        """
        total = 0
        values = 0
        for param in program.all_parameters():
            mask = np.array(paddle.static.global_scope().find_var(
                param.name + "_mask").get_tensor())
            total += np.product(param.shape)
            values += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
        return float(values) / total

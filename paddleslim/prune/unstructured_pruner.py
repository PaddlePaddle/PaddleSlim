import numpy as np
from ..common import get_logger
from ..core import GraphWrapper
import paddle
import copy

__all__ = ["UnstructuredPruner", "UnstructuredPrunerGMP"]


class UnstructuredPruner():
    """
    The unstructure pruner.

    Args:
      - program(paddle.static.Program): The model to be pruned.
      - mode(str): the mode to prune the model, must be selected from 'ratio' and 'threshold'.
      - ratio(float): the ratio to prune the model. Only set it when mode=='ratio'. Default: 0.5.
      - threshold(float): the threshold to prune the model. Only set it when mode=='threshold'. Default: 1e-5.
      - scope(paddle.static.Scope): The scope storing values of all variables. None means paddle.static.global_scope. Default: None.
      - place(CPUPlace | CUDAPlace): The device place used to execute model. None means CPUPlace. Default: None.
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params.
    """

    def __init__(self,
                 program,
                 mode,
                 ratio=0.5,
                 threshold=1e-5,
                 scope=None,
                 place=None,
                 skip_params_func=None,
                 configs=None):
        self.mode = mode
        self.ratio = ratio
        self.threshold = threshold
        assert self.mode in [
            'ratio', 'threshold'
        ], "mode must be selected from 'ratio' and 'threshold'"
        self.scope = paddle.static.global_scope() if scope == None else scope
        self.place = paddle.static.cpu_places()[0] if place is None else place
        if skip_params_func is None: skip_params_func = self._get_skip_params
        self.skip_params = skip_params_func(program)
        self.masks = self._apply_masks(program, self.mask_parameters)

    def mask_parameters(self, parameters, masks, program):
        """
        Update masks and parameters. It is executed before each iteration.
        User can overwrite this function in subclass to implememt different pruning stragies.
        Args:
          - parameters(list<Tensor>): The parameters to be pruned.
          - masks(list<Tensor>): The masks used to keep zero values in parameters.
          - program(paddle.static.Program): The model to add mask op to.
        """
        block = program.global_block()
        for param, mask in zip(parameters, masks):
            block._prepend_op(
                type='elementwise_mul',
                inputs={'X': param,
                        'Y': mask},
                outputs={'Out': param},
                attrs={'axis': -1,
                       'use_mkldnn': False})

    def _apply_masks(self, program, mask_func):
        params = []
        masks = []
        self.no_grad_set = set()

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
            self.no_grad_set.add(param.name + "_mask")

        with paddle.static.program_guard(main_program=program):
            ops = program.global_block().ops
            ori_len = len(ops)
            mask_func(params, masks, program)
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
        Return:
          - threshold(float): a threshold corresponding to the input ratio.
        """
        data = []
        for param in program.all_parameters():
            data.append(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()).flatten())
        data = np.concatenate(data, axis=0)
        threshold = np.sort(np.abs(data))[max(0, int(ratio * len(data) - 1))]
        return threshold

    def sparse_by_layer(self, program):
        """
        The function is used to get the density at each layer, usually called for debuggings.
        
        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - layer_sparse(Dict<string, float>): sparsity for each parameter.
        """
        layer_sparse = {}
        total = 0
        values = 0
        for param in program.all_parameters():
            value = np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
            layer_sparse[param.name] = 1 - float(value) / np.product(
                param.shape)
        return layer_sparse

    def update_threshold(self):
        '''
        Update the threshold after each optimization step in RATIO mode.
        User should overwrite this method to define their own weight importance (Default is based on their absolute values).
        '''
        params_flatten = []
        for param in self.masks:
            if not self._should_prune_param(param):
                continue
            t_param = self.scope.find_var(param).get_tensor()
            v_param = np.array(t_param)
            params_flatten.append(v_param.flatten())
        params_flatten = np.concatenate(params_flatten, axis=0)
        self.threshold = self._partition_sort(params_flatten)

    def _partition_sort(self, params):
        total_len = len(params)
        params_zeros = params[params == 0]
        params_nonzeros = params[params != 0]
        new_ratio = max((self.ratio * total_len - len(params_zeros)),
                        0) / len(params_nonzeros)
        return np.sort(np.abs(params_nonzeros))[max(
            0, int(new_ratio * len(params_nonzeros)) - 1)]

    def update_masks(self):
        for param in self.masks:
            if not self._should_prune_param(param):
                continue
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            v_param[np.abs(v_param) < self.threshold] = 0
            v_mask = (v_param != 0).astype(v_param.dtype)
            t_mask.set(v_mask, self.place)

    def set_static_masks(self):
        for param in self.masks:
            if not self._should_prune_param(param):
                continue
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            v_mask = (v_param != 0).astype(v_param.dtype)
            t_mask.set(v_mask, self.place)

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'threshold':
            pass
        elif self.mode == 'ratio':
            self.update_threshold()
        self.update_masks()

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
        The function is used to get the whole model's sparsity.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.

        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - sparsity(float): the model's sparsity.
        """
        total = 0
        values = 0
        for param in program.all_parameters():
            total += np.product(param.shape)
            values += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
        sparsity = 1 - float(values) / total
        return sparsity

    def _get_skip_params(self, program):
        """
        The function is used to get a set of all the skipped parameters when performing pruning.
        By default, the normalization-related ones will not be pruned.
        Developers could replace it by passing their own function when initializing the UnstructuredPruner instance.

        Args:
          - program(paddle.static.Program): the current model.
        Returns:
          - skip_params(Set<String>): a set of parameters' names.
        """
        skip_params = set()
        graph = GraphWrapper(program)
        for op in graph.ops():
            if 'norm' in op.type() and 'grad' not in op.type():
                for input in op.all_inputs():
                    skip_params.add(input.name())
        return skip_params

    @staticmethod
    def total_sparse_conv1x1(program):
        """
        The function is used to get the model's spasity for all the 1x1 convolutional weights.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.

        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - sparsity(float): the model's sparsity.
        """
        total = 0
        values = 0
        for param in program.all_parameters():
            if not (len(param.shape) == 4 and param.shape[2] == 1 and
                    param.shape[3] == 1):
                continue
            total += np.product(param.shape)
            values += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
        sparsity = 1 - float(values) / total
        return sparsity

    def _should_prune_param(self, param):
        should_prune = param not in self.skip_params
        return should_prune


class UnstructuredPrunerGMP(UnstructuredPruner):
    def __init__(self,
                 program,
                 mode,
                 ratio=0.5,
                 threshold=1e-5,
                 scope=None,
                 place=None,
                 skip_params_func=None,
                 configs=None):
        assert mode == 'ratio', "Mode must be RATIO in GMP pruner."
        assert configs is not None, "Please pass in a valid config dictionary."
        super(UnstructuredPrunerGMP, self).__init__(
            program, mode, ratio, threshold, scope, place, skip_params_func)
        self.stable_iterations = configs.get('stable_iterations')
        self.pruning_iterations = configs.get('pruning_iterations')
        self.tunning_iterations = configs.get('tunning_iterations')
        self.pruning_steps = configs.get('pruning_steps')
        self.initial_ratio = configs.get('initial_ratio')
        self.target_ratio = ratio
        self.cur_iteration = configs.get('resume_iteration')

        self._prepare_training_hyper_parameters()

    def _prepare_training_hyper_parameters(self):
        self.ratios_stack = []
        self.ratio_increment_period = int(self.pruning_iterations /
                                          self.pruning_steps)
        for i in range(self.pruning_steps):
            ratio_tmp = ((i / self.pruning_steps) - 1.0)**3 + 1
            ratio_tmp = ratio_tmp * (self.target_ratio - self.initial_ratio
                                     ) + self.initial_ratio
            self.ratios_stack.append(ratio_tmp)

        stable_steps = int(
            float(self.stable_iterations) / self.pruning_iterations *
            self.pruning_steps)
        tunning_steps = int(
            float(self.tunning_iterations) / self.pruning_iterations *
            self.pruning_steps)
        stable_ratios_stack = [0.0] * stable_steps
        tunning_ratios_stack = [self.target_ratio] * tunning_steps

        self.ratios_stack = stable_ratios_stack + self.ratios_stack + tunning_ratios_stack
        self.ratios_stack.reverse()

        # pop out used ratios to resume training
        for i in range(self.cur_iteration):
            if len(self.
                   ratios_stack) > 0 and i % self.ratio_increment_period == 0:
                self.ratio = self.ratios_stack.pop()

    def step(self):
        ori_ratio = self.ratio
        if self.cur_iteration % self.ratio_increment_period == 0:
            if len(self.ratios_stack) > 0:
                self.ratio = self.ratios_stack.pop()
            else:
                self.ratio = self.target_ratio

        # Update the threshold and masks only when a new ratio has been set.
        # This condition check would save training time dramatically since we only update the threshold by the triger of self.ratio_increment_period.
        if ori_ratio != self.ratio:
            self.update_threshold()
            self.update_masks()
        self.cur_iteration += 1

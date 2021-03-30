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
        self.masks = self._initialize_masks(program)

    def _initialize_masks(self, program):
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

        d_masks = {}
        for _param, _mask in zip(params, masks):
            d_masks[_param.name] = _mask.name
        return d_masks

    def _prune_by_ratio(self, ratios):
        """
        Pruning layers by ratio.
        Args:
          - ratios(dict): The key is name of parameters and the value is the ratio to be pruned.
        """
        for param, ratio in ratios.items():
            assert param in self.masks, "Couldn't find {} in parameters of target program.".format(
                param.name)
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            ori_shape = v_param.shape
            v_param = v_param.flatten()
            pruned_num = int(len(v_param) * ratio)
            v_param[np.abs(v_param).argsort()[:pruned_num]] = 0
            v_param = v_param.reshape(ori_shape)
            v_mask = (v_param != 0).astype(v_param.dtype)
            t_mask.set(v_mask, self.place)
            v_param = np.array(t_param) * np.array(t_mask)
            t_param.set(v_param, self.place)

    def _prune_by_ratio_globally(self, ratios):
        """
        Prune the whole network globally by ratio
        Args:
          - ratios(dict): The key is the name of parameters and the value is the ratio to by pruned
        """
        average_ratio = 0
        total_length = 0
        params_flatten = []
        # get the importance threshold according to the given ratio
        for param, ratio in ratios.items():
            if not 'weight' in param: continue
            assert param in self.masks, "Couldn't find {} in parameters of target program.".format(
                param.name)
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            params_flatten.append(v_param.flatten())
            total_length += int(v_param.size)
            average_ratio = ratio
        params_flatten = np.concatenate(params_flatten, axis=0)
        threshold = np.sort(np.abs(params_flatten))[max(
            0, int(average_ratio * total_length) - 1)]
        # set mask based on the global threshold
        for param, ratio in ratios.items():
            if not 'weight' in param: continue
            assert param in self.masks, "Couldn't find {} in parameters of target program.".format(
                param.name)
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)

            v_param[np.abs(v_param) < threshold] = 0
            v_mask = (v_param != 0).astype(v_param.dtype)
            t_mask.set(v_mask, self.place)
            v_param = np.array(t_param) * np.array(t_mask)
            t_param.set(v_param, self.place)

    def uniform_prune(self, ratio, mode="global"):
        """
        Main function to prune the network given ratios.
        Args:
          - ratio(float): the ratio to set parameters under which to be zeros.
          - mode(str):    the strategy to prune the layers, whether prune by layer or globally according to the ratio. 
                          the global mode is recommanded.
        """
        assert mode in ["layer", "global"]
        ratios = {}
        for param in self.masks:
            ratios[param] = ratio
        if mode == "layer":
            self._prune_by_ratio(ratios)
        elif mode == "global":
            self._prune_by_ratio_globally(ratios)

    def update_params(self):
        """
        Update the parameters given self.masks, usually callen before saving models.
        """
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

    def total_sparse(self, program):
        """
        The function is used to get the whole model's density (1-sparsity).
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

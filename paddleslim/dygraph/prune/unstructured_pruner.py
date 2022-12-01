import numpy as np
import paddle
import logging
from paddleslim.common import get_logger
from paddleslim.prune.unstructured_pruner_utils import *
import copy

__all__ = ["UnstructuredPruner", "GMPUnstructuredPruner"]

_logger = get_logger(__name__, level=logging.INFO)

NORMS_ALL = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'BatchNorm1D',
    'BatchNorm2D', 'BatchNorm3D', 'InstanceNorm1D', 'InstanceNorm2D',
    'InstanceNorm3D', 'SyncBatchNorm', 'LocalResponseNorm'
]


class UnstructuredPruner():
    """
    The unstructure pruner.
    Args:
      - model(Paddle.nn.Layer): The model to be pruned.
      - mode(str): Pruning mode, must be selected from 'ratio' and 'threshold'.
      - threshold(float): The parameters whose absolute values are smaller than the THRESHOLD will be zeros. Default: 0.01
      - ratio(float): The parameters whose absolute values are in the smaller part decided by the ratio will be zeros. Default: 0.55
      - prune_params_type(str): The argument to control which type of ops will be pruned. Currently we only support None (all but norms) or conv1x1_only as input. It acts as a straightforward call to conv1x1 pruning.  Default: None
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params. 
      - local_sparsity(bool): Whether to enable local sparsity. Local sparsity means all the weight matrices have the same sparsity. And the global sparsity only ensures the whole model's sparsity is equal to the passed-in 'ratio'. Default: False
      - sparse_block(Array<Integer>): There must be two integers inside this array. The array defines the shape of the block, the values within which are either sparsified to all zeros or kept original. [1, 1] means unstructured pruning. Default: [1,1]
    """

    def __init__(self,
                 model,
                 mode,
                 threshold=0.01,
                 ratio=0.55,
                 prune_params_type=None,
                 skip_params_func=None,
                 local_sparsity=False,
                 sparse_block=[1, 1]):
        assert mode in ('ratio', 'threshold'
                        ), "mode must be selected from 'ratio' and 'threshold'"
        assert prune_params_type is None or prune_params_type == 'conv1x1_only', "prune_params_type only supports None or conv1x1_only for now."
        if local_sparsity:
            assert mode == 'ratio', "We don't support local_sparsity==True and mode=='threshold' at the same time, please change the inputs accordingly."
        assert len(
            sparse_block
        ) == 2 and sparse_block[0] > 0 and sparse_block[1] > 0 and isinstance(
            sparse_block[0], int
        ) and isinstance(
            sparse_block[1], int
        ), "Please make sure you provide a valid sparse block, in which there are two positive integers."

        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.ratio = ratio
        self.local_sparsity = local_sparsity
        self.thresholds = {}
        self.sparse_block = sparse_block

        # Prority: passed-in skip_params_func > prune_params_type (conv1x1_only) > built-in _get_skip_params
        if skip_params_func is not None:
            skip_params_func = skip_params_func
        elif prune_params_type == 'conv1x1_only':
            skip_params_func = self._get_skip_params_conv1x1
        elif skip_params_func is None:
            skip_params_func = self._get_skip_params

        self.skip_params = skip_params_func(model)
        self._apply_masks()

    def mask_parameters(self, param, mask):
        """
        Update masks and parameters. It is executed to each layer before each iteration.
        User can overwrite this function in subclass to implememt different pruning stragies.
        Args:
          - parameters(list<Tensor>): The parameters to be pruned.
          - masks(list<Tensor>): The masks used to keep zero values in parameters.
        """
        param_tmp = param * mask
        param_tmp.stop_gradient = True
        paddle.assign(param_tmp, output=param)

    def _apply_masks(self):
        self.masks = {}
        for name, sub_layer in self.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                tmp_array = np.ones(param.shape, dtype=np.float32)
                mask_name = "_".join([param.name.replace(".", "_"), "mask"])
                if mask_name not in sub_layer._buffers:
                    print(f"target type: {type(paddle.to_tensor(tmp_array))}")
                    sub_layer.register_buffer(mask_name,
                                              paddle.to_tensor(tmp_array))
                self.masks[param.name] = sub_layer._buffers[mask_name]
        for name, sub_layer in self.model.named_sublayers():
            sub_layer.register_forward_pre_hook(self._forward_pre_hook)

    def update_threshold(self):
        '''
        Update the threshold after each optimization step.
        User should overwrite this method togther with self.mask_parameters()
        '''
        params_flatten = []
        for name, sub_layer in self.model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                if param.name in self.skip_params:
                    continue
                t_param = param.value().get_tensor()
                v_param = np.array(t_param)

                if (self.sparse_block[0] * self.sparse_block[1] / v_param.size
                        >= BLOCK_SPARSE_ACCURATE_THRESHOLD):
                    print(
                        "Your sparse block size {} might be too large for the param {} with shape {}, the sparsity of this param might not be precise. Please decrease your sparse block size if possible. Currently, sparse_block[0] ({}) X sparse_block[1] ({}) / weight_count ({}) >= {}".
                        format(self.sparse_block, param, v_param.shape,
                               self.sparse_block[0], self.sparse_block[1],
                               v_param.size, BLOCK_SPARSE_ACCURATE_THRESHOLD))
                v_param = cal_mxn_avg_matrix(
                    v_param, m=self.sparse_block[0], n=self.sparse_block[1])

                if self.local_sparsity:
                    flatten_v_param = v_param.flatten()
                    cur_length = flatten_v_param.size
                    cur_threshold = np.sort(np.abs(flatten_v_param))[max(
                        0, round(self.ratio * cur_length) - 1)].item()
                    self.thresholds[param.name] = cur_threshold
                else:
                    params_flatten.append(v_param.flatten())
        if not self.local_sparsity:
            params_flatten = np.concatenate(params_flatten, axis=0)
            total_length = params_flatten.size
            self.threshold = np.sort(np.abs(params_flatten))[max(
                0, round(self.ratio * total_length) - 1)].item()

    def _update_masks(self):
        for name, sub_layer in self.model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                if param.name in self.skip_params:
                    continue
                mask = self.masks.get(param.name)
                v_param = np.array(param.value().get_tensor())
                v_param_avg = cal_mxn_avg_matrix(
                    v_param, m=self.sparse_block[0], n=self.sparse_block[1])
                if self.local_sparsity:
                    bool_tmp = (abs(v_param_avg) >= self.thresholds[param.name])
                else:
                    bool_tmp = (abs(v_param_avg) >= self.threshold)
                paddle.assign(bool_tmp, output=mask)

    def set_static_masks(self):
        for name, sub_layer in self.model.named_sublayers():
            if not self._should_prune_layer(sub_layer): continue
            for param in sub_layer.parameters(include_sublayers=False):
                mask = self.masks.get(param.name)
                bool_tmp = (paddle.abs(param) != 0.0)
                paddle.assign(bool_tmp, output=mask)

    def summarize_weights(self, model, ratio=0.1):
        """
        The function is used to get the weights corresponding to a given ratio
        when you are uncertain about the threshold in __init__() function above.
        For example, when given 0.1 as ratio, the function will print the weight value,
        the abs(weights) lower than which count for 10% of the total numbers.
        Args:
          - model(paddle.nn.Layer): The model which have all the parameters.
          - ratio(float): The ratio illustrated above.
        Return:
          - threshold(float): a threshold corresponding to the input ratio.
        """
        data = []
        for name, sub_layer in model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                data.append(np.array(param.value().get_tensor()).flatten())
        data = np.concatenate(data, axis=0)
        threshold = np.sort(np.abs(data))[max(0, int(ratio * len(data) - 1))]
        return threshold

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'ratio':
            self.update_threshold()
            self._update_masks()
        elif self.mode == 'threshold':
            self._update_masks()

    def _forward_pre_hook(self, layer, input):
        if not self._should_prune_layer(layer):
            return input
        for param in layer.parameters(include_sublayers=False):
            mask = self.masks.get(param.name)
            self.mask_parameters(param, mask)
        return input

    def update_params(self):
        """
        Update the parameters given self.masks, usually called before saving models and evaluation step during training. 
        If you load a sparse model and only want to inference, no need to call the method.
        """
        for name, sub_layer in self.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                mask = self.masks.get(param.name)
                param_tmp = param * mask
                param_tmp.stop_gradient = True
                paddle.assign(param_tmp, output=param)

    @staticmethod
    def total_sparse(model):
        """
        This static function is used to get the whole model's sparsity.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        
        Args:
          - model(paddle.nn.Layer): The sparse model.
        Returns:
          - ratio(float): The model's sparsity.
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = 1 - float(values) / total
        return ratio

    @staticmethod
    def total_sparse_conv1x1(model):
        """
        This static function is used to get the partial model's sparsity in terms of conv1x1 layers.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        
        Args:
          - model(paddle.nn.Layer): The sparse model.
        Returns:
          - ratio(float): The model's sparsity.
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                cond = len(param.shape) == 4 and param.shape[
                    2] == 1 and param.shape[3] == 1
                if not cond: continue
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = 1 - float(values) / total
        return ratio

    def _get_skip_params(self, model):
        """
        This function is used to check whether the given model's layers are valid to be pruned. 
        Usually, the convolutions are to be pruned while we skip the normalization-related parameters and bias.
        Deverlopers could replace this function by passing their own when initializing the UnstructuredPuner instance.

        Args:
          - model(Paddle.nn.Layer): the current model waiting to be checked.
        Return:
          - skip_params(set<String>): a set of parameters' names
        """
        skip_params = set()
        for _, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                skip_params.add(sub_layer.full_name())
            # exclude bias whose shape is like (n,)
            for param in sub_layer.parameters(include_sublayers=False):
                if len(param.shape) == 1: skip_params.add(param.name)
        return skip_params

    def _get_skip_params_conv1x1(self, model):
        skip_params = set()
        for _, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                skip_params.add(sub_layer.full_name())
            for param in sub_layer.parameters(include_sublayers=False):
                # exclude bias whose shape is like (n,)
                if len(param.shape) == 1: skip_params.add(param.name)
                cond = len(param.shape) == 4 and param.shape[
                    2] == 1 and param.shape[3] == 1
                if not cond: skip_params.add(param.name)
        return skip_params

    def _should_prune_layer(self, layer):
        should_prune = layer.full_name() not in self.skip_params
        return should_prune


class GMPUnstructuredPruner(UnstructuredPruner):
    """
    The unstructure pruner using GMP training strategy (Gradual Magnitute Pruning). In this subclass of UnstructuredPruner, most methods are inheritated apart from the step(), since we add some ratio increment logics here.
    Conceptually, the algorithm divide the training into three phases: stable, pruning and tuning. And the ratio is increasing from initial_ratio gradually and nonlinearly w.r.t. the training epochs/iterations.

    Args:
      - model(Paddle.nn.Layer): The model to be pruned.
      - ratio(float): The parameters whose absolute values are in the smaller part decided by the ratio will be zeros. Default: 0.55
      - prune_params_type(str): The argument to control which type of ops will be pruned. Currently we only support None (all but norms) or conv1x1_only as input. It acts as a straightforward call to conv1x1 pruning.  Default: None
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params. 
      - local_sparsity(bool): Whether to enable local sparsity. Local sparsity means all the weight matrices have the same sparsity. And the global sparsity only ensures the whole model's sparsity is equal to the passed-in 'ratio'. Default: False
      - sparse_block(Array<Integer>): There must be two integers inside this array. The array defines a block, the values within which are either sparsified to all zeros or kept original. [1, 1] means unstructured pruning. Default: [1,1]
      - configs(Dict): The dictionary contains all the configs for GMP pruner. Default: None

        .. code-block:: python
               
               {'stable_iterations': int} # the duration of stable phase in terms of global iterations
               {'pruning_iterations': int} # the duration of pruning phase in terms of global iterations
               {'tunning_iterations': int} # the duration of tunning phase in terms of global iterations
               {'resume_iteration': int} # the start timestamp you want to train from, in terms if global iteration
               {'pruning_steps': int} # the total times you want to increase the ratio
               {'initial_ratio': float} # the initial ratio value
        
        ..

    """

    def __init__(self,
                 model,
                 ratio=0.55,
                 prune_params_type=None,
                 skip_params_func=None,
                 local_sparsity=False,
                 sparse_block=[1, 1],
                 configs=None):

        assert configs is not None, "Configs must be passed in for GMP pruner."
        super(GMPUnstructuredPruner, self).__init__(
            model, 'ratio', 0.0, ratio, prune_params_type, skip_params_func,
            local_sparsity, sparse_block)
        self.stable_iterations = configs.get('stable_iterations')
        self.pruning_iterations = configs.get('pruning_iterations')
        self.tunning_iterations = configs.get('tunning_iterations')
        self.pruning_steps = configs.get('pruning_steps')
        self.initial_ratio = configs.get('initial_ratio')
        self.ratio = 0.0
        self.target_ratio = ratio
        self.cur_iteration = configs.get('resume_iteration')

        assert self.pruning_iterations / self.pruning_steps > 10, "To guarantee the performance of GMP pruner, pruning iterations must be larger than pruning steps by a margin."
        self._need_prune_once = False
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
            self._need_prune_once = True
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
        if ori_ratio != self.ratio or self._need_prune_once:
            self.update_threshold()
            self._update_masks()
            self._need_prune_once = False
        self.cur_iteration += 1

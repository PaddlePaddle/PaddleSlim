import numpy as np
import paddle
import logging
from paddleslim.common import get_logger

_logger = get_logger(__name__, level=logging.INFO)


class UnstructurePruner():
    """
    The unstructure pruner.
    Args:
      - model(Paddle.Model): The model to be pruned.
      - mode(ratio | threshold): Pruning mode, whether by the given ratio or threshold.
      - threshold(float): The parameters whose absolute values are smaller than the THRESHOLD will be zeros. Default: None
      - ratio(float): The parameters whose absolute values are in the smaller part decided by the ratio will be zeros. Default: None
    """

    def __init__(self, model, mode, threshold=0.01, ratio=0.3):
        assert mode in ('ratio', 'threshold'
                        ), "mode must be selected from 'ratio' and 'threshold'"
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.ratio = ratio
        self._apply_masks()

    def mask_parameters(self, param, mask):
        """
        Update masks and parameters. It is executed to each layer before each iteration.
        User can overwrite this function in subclass to implememt different pruning stragies.
        Args:
          - parameters(list<Tensor>): The parameters to be pruned.
          - masks(list<Tensor>): The masks used to keep zero values in parameters.
        """
        bool_tmp = (paddle.abs(param) >= self.threshold)
        paddle.assign(mask * bool_tmp, output=mask)
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
            if not self.should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                t_param = param.value().get_tensor()
                v_param = np.array(t_param)
                params_flatten.append(v_param.flatten())
        params_flatten = np.concatenate(params_flatten, axis=0)
        total_length = params_flatten.size
        self.threshold = np.sort(np.abs(params_flatten))[max(
            0, round(self.ratio * total_length) - 1)].item()

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'ratio':
            self.update_threshold()
        elif self.mode == 'threshold':
            return

    def _forward_pre_hook(self, layer, input):
        if not self.should_prune_layer(layer):
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
        This static function is used to get the whole model's density (1-sparsity).
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        
        Args:
          - model(Paddle.Model): The sparse model.
        Returns:
          - ratio(float): The model's density.
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = float(values) / total
        return ratio

    def should_prune_layer(self, layer):
        """
        This function is used to check whether the given sublayer is valid to be pruned. 
        Usually, the convolutions are to be pruned while we skip the BN-related parameters.
        Developers could overwrite this function to define their own according to their model architectures.

        Args:
          - layer(): the sublayer waiting to be checked.
        Return:
          - should_prune(bool): whether the sublayer should be pruned or not.
        """
        if type(layer).__name__.split('.')[-1] in paddle.nn.norm.__all__:
            should_prune = False
        else:
            should_prune = True
        return should_prune

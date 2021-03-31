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
      - weights_keywords(List<String>): A list of string containing the keywords, meaning if this keyword is a substring of the parameter name, the parameter will be pruned. This argument could be used to ignore BN if you don't want to prune them. 
    """

    def __init__(self,
                 model,
                 mode,
                 threshold=None,
                 ratio=None,
                 weights_keywords=['conv']):
        assert mode in ('ratio', 'threshold'
                        ), "mode must be selected from 'ratio' and 'threshold'"
        if mode == 'ratio' and ratio is None:
            _logger.info(
                "The ratio mode is selected without setting the RATIO, default: 0.3"
            )
            ratio = 0.3
            threshold = 0.0
        if mode == 'threshold' and threshold is None:
            _logger.info(
                "The threshold mode is selected without setting the THRESHOLD, default: 0.01"
            )
            threshold = 0.01
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.ratio = ratio
        self.weights_keywords = weights_keywords
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
            for param in sub_layer.parameters(include_sublayers=False):
                if not 'conv' in param.name: continue
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
        for param in layer.parameters(include_sublayers=False):
            mask = self.masks.get(param.name)
            for word in self.weights_keywords:
                if word in param.name:
                    self.mask_parameters(param, mask)
                    return input
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
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = float(values) / total
        return ratio

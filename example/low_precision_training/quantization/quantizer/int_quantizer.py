import paddle
from paddle import tensor as T
from ..quantization_utils import custom_round

class IntQuantizer:
    r"""Integer quantizer.
    Args:
        bitwidth (int): The bit-width for quantization
        scaling (str): quantization scale type. Support: ``fixed``, ``learned``, ``dynamic``
        dims (None, int or tuple of ints): the dimensions to be assigned with a set of scales
        observer (str): quantization metric to calculate scale. Support: ``minmax``, ``mse``, ``cosine``
        tag (None or str): quantization tag
    """
    def __init__(self, bitwidth, signed=None, scaling="fixed", dims=None, observer="minmax", stochastic=False):
        self.bitwidth = bitwidth
        self.signed = signed
        assert scaling in ["fixed", "learned", "dynamic"]
        self.scaling = scaling
        if dims is None:
            self.dims = None
        elif isinstance(dims, int):
            self.dims = (dims,)
        elif isinstance(dims, tuple) or isinstance(dims, list):
            self.dims = tuple(dims)
        elif isinstance(dims, str):
            self.dims = self._parse_readable_dims(dims)
        else:
            raise TypeError(f'dims must be None, int, list or tuple, but got {type(dims)}')
        assert observer in ["minmax", "mse", "cosine"]
        self.observer = observer
        self.stochastic = stochastic

        if self.scaling == "fixed":
            self.fixed_scale = None

        if self.signed is True:
            self.lower_bound = -2 ** (self.bitwidth - 1)
            self.upper_bound = -self.lower_bound - 1
        elif self.signed is False:
            self.upper_bound = (1 << self.bitwidth) - 1
            self.lower_bound = 0
        else:
            self.upper_bound = self.lower_bound = None

    def _if_is_signed(self, tensor):
        if self.signed is None:
            if tensor.min() < 0:
                self.signed = True
                self.lower_bound = -2 ** (self.bitwidth - 1)
                self.upper_bound = -self.lower_bound - 1
            else:
                self.signed = False
                self.upper_bound = (1 << self.bitwidth) - 1
                self.lower_bound = 0

    def _parse_readable_dims(self, str_dims: str):
        dims = []
        dim_map = {'n': 0, 'c': 1, 'h': 2, 'w': 3}
        str_dims = str_dims.lower()
        if str_dims in ['per-layer', 'per-tensor', 'none']:
            return None
        str_dims = list(str_dims)
        try:
            for str_dim in str_dims:
                dims.append(dim_map[str_dim])
        except KeyError:
            ex = AttributeError(f"quantization granularity only support NCHW, but got '{str_dim}'")
            raise ex
        except Exception as e:
            raise e
        else:
            return tuple(dims)

    def set_scale(self, scale):
        assert self.scaling == "fixed"
        self.fixed_scale = scale

    def calculate_quantization_scale(self, tensor, eps=1e-6, max_iter=50):
        if self.observer == "minmax":
            return self.minmax_quantize_with_dynamic_scale(tensor)[1]
        elif self.observer == "mse":
            return self.mse_quantize_with_dynamic_scale(tensor, eps, max_iter)[1]
        else:
            assert False

    def movedim_to_perm(self, dims):
        if dims == (0,):
            return (0,1,2,3), (0,1,2,3)
        elif dims == (1,):
            return (1,0,2,3), (1,0,2,3)
        elif dims == (0,1):
            return (0,1,2,3), (0,1,2,3)
        elif dims == (0,2,3):
            return (0,2,3,1), (0,3,1,2)
        else:
            raise TypeError(f'dims {type(dims)} is unexpected!')



    def quantize_with_fixed_scale(self, tensor):
        self._if_is_signed(tensor)

        if self.dims is None: 
            # per-layer quantization
            quantized_tensor = custom_round(tensor / self.fixed_scale, self.stochastic).clip(self.lower_bound, self.upper_bound) * self.fixed_scale
        else:
            tensor_shape = tensor.shape
            source = self.dims
            dest = tuple(range(len(source)))

            num_scales = 1
            for i in source:
                num_scales *= tensor_shape[i]

            permute_tensor = paddle.transpose(tensor, self.movedim_to_perm(self.dims)[0]).contiguous()
            permute_tensor_shape = permute_tensor.shape
            flatten_tensor = permute_tensor.reshape([num_scales, -1])

            flatten_tensor_q = custom_round(flatten_tensor / self.fixed_scale[:, None], self.stochastic).clip(self.lower_bound, self.upper_bound)
            quantized_tensor = (flatten_tensor_q * self.fixed_scale[:, None]).reshape(permute_tensor_shape).transpose(self.movedim_to_perm(self.dims)[1]).contiguous()

        return quantized_tensor

    def quantize_with_learned_scale(self, tensor):
        pass
    
    def minmax_quantize_with_dynamic_scale(self, tensor):
        self._if_is_signed(tensor)

        if self.dims is None: 
            # per-layer quantization
            alpha = paddle.abs(tensor.flatten()).max() / self.upper_bound
            alpha = paddle.clip(alpha, min=1e-12)
            quantized_tensor = custom_round(tensor / alpha, self.stochastic).clip(self.lower_bound, self.upper_bound) * alpha
            
        else:
            tensor_shape = tensor.shape
            source = self.dims
            dest = tuple(range(len(source)))

            num_scales = 1
            for i in source:
                num_scales *= tensor_shape[i]
                
            permute_tensor = paddle.transpose(tensor, self.movedim_to_perm(self.dims)[0]).contiguous()
            permute_tensor_shape = permute_tensor.shape

            # print("permute_tensor.shape: ", permute_tensor.shape) # [64, 64, 3, 3]
            flatten_tensor = permute_tensor.reshape([num_scales, -1])

            # print(flatten_tensor.shape)  # [64, 576]
            alpha = flatten_tensor.abs().max(axis=1) / self.upper_bound
            # print(alpha.shape)
            alpha = paddle.clip(alpha, min=1e-12)  ####

            # print(alpha.shape)  # []
            # print(flatten_tensor.shape) # [36864]
            # print(1, flatten_tensor.abs().max(axis=1).shape)
            # print(flatten_tensor.abs().max(axis=1)[0].shape)

            flatten_tensor_q = custom_round(flatten_tensor / alpha[:, None], self.stochastic).clip(self.lower_bound, self.upper_bound)

            quantized_tensor = (flatten_tensor_q * alpha[:, None]).reshape(permute_tensor_shape).transpose(self.movedim_to_perm(self.dims)[1]).contiguous()

        return quantized_tensor, alpha

    def mse_quantize_with_dynamic_scale(self, tensor, eps=1e-6, max_iter=50):
        self._if_is_signed(tensor)
        
        if self.dims is None: 
            # per-layer quantization
            flatten_tensor = tensor.flatten()
            alpha = flatten_tensor.abs().max() / self.upper_bound
            alpha_old = alpha * 1.2
            for it in range(max_iter):
                flatten_tensor_q = (flatten_tensor / alpha).round().clip(self.lower_bound, self.upper_bound)
                alpha_old = alpha
                alpha = flatten_tensor.dot(flatten_tensor_q) / flatten_tensor_q.dot(flatten_tensor_q)
                if paddle.abs((alpha - alpha_old) / alpha).item() <= eps:
                    break
            quantized_tensor = custom_round(tensor / alpha, self.stochastic).clip(self.lower_bound, self.upper_bound) * alpha
        else:
            tensor_shape = tensor.shape
            source = self.dims
            dest = tuple(range(len(source)))

            num_scales = 1
            for i in source:
                num_scales *= tensor_shape[i]

            permute_tensor = paddle.transpose(tensor, self.movedim_to_perm(self.dims)[0]).contiguous()
            permute_tensor_shape = permute_tensor.shape

            # 使用 reshape 替代 view
            flatten_tensor = permute_tensor.reshape([num_scales, -1])

            alpha = paddle.abs(flatten_tensor).max(axis=1) / self.upper_bound
            alpha_old = alpha * 1.2

            for it in range(max_iter):
                flatten_tensor_q = (flatten_tensor / alpha[:, None]).round().clip(self.lower_bound, self.upper_bound)
                alpha_old = alpha
                alpha = (flatten_tensor * flatten_tensor_q).sum(axis=1) / (flatten_tensor_q * flatten_tensor_q).sum(axis=1)
                if ((alpha - alpha_old).norm() / alpha_old.norm()).item() <= eps:
                    break

            flatten_tensor_q = custom_round(flatten_tensor / alpha[:, None], self.stochastic).clip(self.lower_bound, self.upper_bound)
            quantized_tensor = (flatten_tensor_q * alpha[:, None]).reshape(permute_tensor_shape).transpose(self.movedim_to_perm(self.dims)[1]).contiguous()

            # 使用 transpose 替代 movedim
            # quantized_tensor = paddle.transpose(quantized_tensor, perm=[*source, *[j for j in range(len(permute_tensor_shape)) if j not in dest]])

        return quantized_tensor, alpha


    def __call__(self, tensor):
        if self.scaling == "fixed":
            return self.quantize_with_fixed_scale(tensor)
        elif self.scaling == "learned":
            return self.quantize_with_learned_scale(tensor)
        elif self.scaling == "dynamic":
            if self.observer == "minmax":  # here
                return self.minmax_quantize_with_dynamic_scale(tensor)[0]
            elif self.observer == "mse":
                return self.mse_quantize_with_dynamic_scale(tensor, max_iter=1)[0]
            else:
                assert False
        else:
            assert False

def int_quantizer(bitwidth, signed=None, scaling="fixed", dims=None, observer="minmax", stochastic=False):
    return IntQuantizer(bitwidth, signed, scaling, dims, observer, stochastic)

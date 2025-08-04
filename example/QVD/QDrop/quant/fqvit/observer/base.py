import paddle

class BaseObserver:
    def __init__(self, module_type, bit_type, calibration_mode):
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = paddle.float32.eps  # 使用Paddle的eps值

    def reshape_tensor(self, v):
        if not isinstance(v, paddle.Tensor):  # 检查是否为Paddle的Tensor
            v = paddle.to_tensor(v)  # 转换为Paddle的Tensor
        v = v.detach()  # Paddle中没有直接的detach()，需要使用stop_gradient()来避免梯度传播
        if self.module_type in ["conv_weight", "linear_weight"]:
            v = v.reshape([v.shape[0], -1])  # 重新调整形状
        elif self.module_type == "activation":
            if len(v.shape) == 4:
                v = paddle.transpose(v, perm=[0, 2, 3, 1])  # 转置操作
            v = v.reshape([-1, v.shape[-1]])
            v = paddle.transpose(v, perm=[1, 0])  # 转置操作
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError

import paddle


class QuantizedLayerNorm(paddle.nn.Layer):

    def __init__(self, org_module):
        super(QuantizedLayerNorm, self).__init__()
        self.normalized_shape = org_module.normalized_shape
        self.eps = org_module.eps
        self.elementwise_affine = org_module.elementwise_affine
        self.weight = org_module.weight
        self.bias = org_module.bias

    def forward(self, input):
        return paddle.nn.functional.layer_norm(x=input, normalized_shape=
            self.normalized_shape, weight=self.weight, bias=self.bias,
            epsilon=self.eps)

    def extra_repr(self) ->str:
        return (
            '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'
            .format(**self.__dict__))


class Identity(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.migrate = False
        self.migrate_scale = None
        self.migrate_bias = None

    def set_migrate(self, state):
        if self.migrate_scale is None:
            self.migrate = False
        else:
            self.migrate = state

    def set_migrate_scale(self, migrate_scale):
        self.migrate_scale = migrate_scale
        self.migrate = True

    def set_migrate_bias(self, migrate_bias):
        self.migrate_bias = migrate_bias
        self.migrate = True

    def forward(self, X):
        if self.migrate:
            if self.migrate_bias is not None:
                X = X - self.migrate_bias
            if self.migrate_scale is not None:
                X /= self.migrate_scale
        return X

import paddle


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module._layers
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = param.clone().detach()

    def update(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module._layers
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = ((
                    1. - self.mu) * param + self.mu * paddle.to_tensor(self.shadow[name])).detach()

    def ema(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module._layers
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                param.stop_gradient = True
                param[:] = self.shadow[name]
                param.stop_gradient = False

    def ema_copy(self, module):
        if isinstance(module, paddle.DataParallel):
            inner_module = module._layers
            module_copy = type(inner_module)(
                inner_module.config)
            module_copy.set_state_dict(inner_module.state_dict())
            module_copy = paddle.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config)
            module_copy.set_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def set_state_dict(self, state_dict):
        self.shadow = state_dict

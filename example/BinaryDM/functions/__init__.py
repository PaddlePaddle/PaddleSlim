import paddle.nn as nn
import paddle.optimizer as optim


def get_optimizer(config, parameters):
    clip = nn.ClipGradByNorm(clip_norm=config.optim.grad_clip)
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters=parameters, learning_rate=config.optim.lr, weight_decay=config.optim.weight_decay,
                          beta1=config.optim.beta1, beta2=0.999, 
                          epsilon=config.optim.eps, grad_clip=clip)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters=parameters, learning_rate=config.optim.lr, weight_decay=config.optim.weight_decay, grad_clip=clip)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters=parameters, learning_rate=config.optim.lr, momentum=0.9, grad_clip=clip)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

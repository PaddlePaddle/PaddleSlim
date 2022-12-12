import paddle.nn as nn
import math


class MyNetwork(nn.Layer):
    def forward(self, x):
        raise NotImplementedError

    def feature_extract(self, x):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    def cfg2params(self, cfg):
        raise NotImplementedError

    def cfg2flops(self, cfg):
        raise NotImplementedError

    def init_model(self, model_init, init_div_groups=False):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                if model_init == 'he_fout':
                    n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                    if init_div_groups:
                        n /= m.groups
                    initializer.normal_(m.weight, 0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    initializer.normal_(m.weight, 0, math.sqrt(2. / n))
                elif model_init == 'xavier_normal':
                    initializer.xavier_normal_(m.weight)
                elif model_init == 'xavier_uniform':
                    initializer.xavier_uniform_(m.weight)
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.stop_gradient = True
                m.bias.stop_gradient = True
                m.weight.fill_(1)
                m.bias.zero_()
                m.weight.stop_gradient = False
                m.bias.stop_gradient = False
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.shape[1])
                initializer.uniform_(m.weight, -stdv, stdv)
                if m.bias is not None:
                    m.bias.stop_gradient = True
                    m.bias.zero_()
                    m.bias.stop_gradient = False
            elif isinstance(m, nn.BatchNorm1D):
                m.weight.fill_(1)
                m.bias.zero_()

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self):
        return self.get_parameters()

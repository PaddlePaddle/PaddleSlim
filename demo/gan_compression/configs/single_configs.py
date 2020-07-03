"""Based on https://github.com/mit-han-lab/gan-compression """


class SingleConfigs:
    def __init__(self, config):
        self.attributes = ['n_channels']
        self.configs = [config]

    def sample(self):
        return self.configs[0]

    def largest(self):
        return self.configs[0]

    def all_configs(self):
        for config in self.configs:
            yield config

    def __call__(self, name):
        assert name in ('largest', 'smallest')
        if name == 'largest':
            return self.largest()
        elif name == 'smallest':
            return self.smallest()
        else:
            raise NotImplementedError

    def __str__(self):
        ret = ''
        for attr in self.attributes:
            ret += 'attr: %s\n' % str(getattr(self, attr))
        return ret

    def __len__(self):
        return len(self.configs)

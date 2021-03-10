"""Based on https://github.com/mit-han-lab/gan-compression """

import random


class ResnetConfigs:
    def __init__(self, n_channels):
        self.attributes = ['n_channels']
        self.n_channels = n_channels

    def sample(self):
        ret = {}
        ret['channel'] = []
        for n_channel in self.n_channels:
            ret['channel'].append(random.choice(n_channel))
        return ret

    def largest(self):
        ret = {}
        ret['channel'] = []
        for n_channel in self.n_channels:
            ret['channel'].append(max(n_channel))
        return ret

    def smallest(self):
        ret = {}
        ret['channel'] = []
        for n_channel in self.n_channels:
            ret['channel'].append(min(n_channel))
        return ret

    def all_configs(self):
        def yield_channels(i):
            if i == len(self.n_channels):
                yield []
                return
            for n in self.n_channels[i]:
                for after_channels in yield_channels(i + 1):
                    yield [n] + after_channels

        for channels in yield_channels(0):
            yield {'channel': channels}

    def __call__(self, name):
        assert name in ('largest', 'smallest')
        if name == 'largest':
            return self.largest()
        elif name == 'smallest':
            return self.smallest()
        else:
            raise NotImplementedError

    def __len__(self):
        ret = 1
        for n_channel in self.n_channels:
            ret *= len(n_channel)


def get_configs(config_name):
    if config_name == 'channels-48':
        return ResnetConfigs(
            n_channels=[[48, 32], [48, 32], [48, 40, 32], [48, 40, 32],
                        [48, 40, 32], [48, 40, 32], [48, 32, 24, 16],
                        [48, 32, 24, 16]])
    elif config_name == 'channels-32':
        return ResnetConfigs(
            n_channels=[[32, 24, 16], [32, 24, 16], [32, 24, 16], [32, 24, 16],
                        [32, 24, 16], [32, 24, 16], [32, 24, 16], [32, 24, 16]])
    elif config_name == 'debug':
        return ResnetConfigs(n_channels=[[48, 32], [48, 32], [48, 40, 32],
                                         [48, 40, 32], [48, 40, 32],
                                         [48, 40, 32], [48], [48]])
    elif config_name == 'test':
        return ResnetConfigs(n_channels=[[8], [6, 8], [6, 8], [8], [8], [8],
                                         [8], [8]])
    else:
        raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)

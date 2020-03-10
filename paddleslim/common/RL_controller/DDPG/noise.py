__all__ = ['AdaptiveNoiseSpec']


class AdaptiveNoiseSpec(object):
    def __init__(self):
        self.stdev_curr = 1.0

    def reset(self):
        self.stdev_curr = 1.0

    def update(self, action_dist):
        if action_dist > 1e-2:
            self.stdev_curr /= 1.03
        else:
            self.stdev_curr *= 1.03

"""
class LRPolicy
"""
import math

__all__ = ["LRPolicy"]


class LRPolicy:
    """
    learning rate policy
    """

    def __init__(self, lr, n_epochs, lr_policy="multi_step"):
        self.lr_policy = lr_policy
        self.params_dict = {}
        self.n_epochs = n_epochs
        self.base_lr = lr
        self.lr = lr

    def set_params(self, params_dict=None):
        """
        set parameters of lr policy
        """
        if self.lr_policy == "multi_step":
            """
            params: decay_rate, step
            """
            self.params_dict['decay_rate'] = params_dict['decay_rate']
            self.params_dict['step'] = sorted(params_dict['step'])
            if max(self.params_dict['step']) <= 1:
                new_step_list = []
                for ratio in self.params_dict['step']:
                    new_step_list.append(int(self.n_epochs * ratio))
                self.params_dict['step'] = new_step_list

        elif self.lr_policy == "step":
            """
            params: end_lr, step
            step: lr = base_lr*gamma^(floor(iter/step))
            """
            self.params_dict['end_lr'] = params_dict['end_lr']

            self.params_dict['step'] = params_dict['step']
            max_iter = math.floor((self.n_epochs - 1.0) /
                                  self.params_dict['step'])

            if self.params_dict['end_lr'] == -1:
                self.params_dict['gamma'] = params_dict['decay_rate']
            else:
                self.params_dict['gamma'] = math.pow(
                    self.params_dict['end_lr'] / self.base_lr, 1. / max_iter)

        elif self.lr_policy == "linear":
            """
            params: end_lr, step
            """
            self.params_dict['end_lr'] = params_dict['end_lr']
            self.params_dict['step'] = params_dict['step']

        elif self.lr_policy == "exp":
            """
            params: end_lr
            exp: lr = base_lr*gamma^iter
            """
            self.params_dict['end_lr'] = params_dict['end_lr']
            self.params_dict['gamma'] = math.pow(
                self.params_dict['end_lr'] / self.base_lr, 1. / (self.n_epochs - 1))

        elif self.lr_policy == "inv":
            """
            params: end_lr
            inv: lr = base_lr*(1+gamma*iter)^(-power)
            """
            self.params_dict['end_lr'] = params_dict['end_lr']
            self.params_dict['power'] = params_dict['power']
            self.params_dict['gamma'] = (math.pow(
                self.base_lr / self.params_dict['end_lr'],
                1. / self.params_dict['power']) - 1.) / (self.n_epochs - 1.)

        elif self.lr_policy == "const":
            """
            no params
            const: lr = base_lr
            """
            self.params_dict = None

        else:
            assert False, "invalid lr_policy" + self.lr_policy

    def get_lr(self, epoch):
        """
        get current learning rate
        """
        if self.lr_policy == "multi_step":
            gamma = 0
            for step in self.params_dict['step']:
                if epoch + 1.0 > step:
                    gamma += 1
            lr = self.base_lr * math.pow(self.params_dict['decay_rate'], gamma)

        elif self.lr_policy == "step":
            lr = self.base_lr * \
                math.pow(self.params_dict['gamma'], math.floor(
                    epoch * 1.0 / self.params_dict['step']))

        elif self.lr_policy == "linear":
            k = (self.params_dict['end_lr'] - self.base_lr) / \
                math.ceil(self.n_epochs / self.params_dict['step'])

            lr = k * math.ceil((epoch + 1) /
                               self.params_dict['step']) + self.base_lr

        elif self.lr_policy == "inv":
            lr = self.base_lr * \
                math.pow(
                    1 + self.params_dict['gamma'] * epoch, -self.params_dict['power'])

        elif self.lr_policy == "exp":
            # power = math.floor((epoch + 1) / self.params_dict['step'])
            # lr = self.base_lr * math.pow(self.params_dict['gamma'], power)
            lr = self.base_lr * math.pow(self.params_dict['gamma'], epoch)

        elif self.lr_policy == "const":
            lr = self.base_lr

        else:
            assert False, "invalid lr_policy: " + self.lr_policy
        self.lr = lr
        return lr

import os
import paddle.fluid as fluid


class BaseModel(fluid.dygraph.Layer):
    @staticmethod
    def add_special_cfgs(parser):
        pass

    def set_input(self, inputs):
        pass

    def setup(self):
        self.load_network()

    def load_network(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.args, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path)

    def save_network(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s' % (epoch, name)
                save_path = os.path.join(self.args.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                fluid.save_dygraph(net.state_dict(), save_path)

    def forward(self):
        pass

    def optimize_parameter(self):
        pass

    def get_current_loss(self):
        loss_dict = {}
        for name in self.loss_names:
            if not hasattr(self, 'loss_' + name):
                continue
            key = name
            loss_dict[key] = float(getattr(self, 'loss_' + name))
        return loss_dict

    def get_current_lr(self):
        raise NotImplementedError

    def set_stop_gradient(self, nets, stop_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = stop_grad

    def evaluate_model(self):
        pass

    def profile(self):
        pass

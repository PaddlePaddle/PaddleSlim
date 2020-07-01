import os
import time
import logging

import paddle.fluid as fluid
from dataset.data_loader import create_data
from utils.get_args import configs


class gan_compression:
    def __init__(self, cfgs, **kwargs):
        self.cfgs = cfgs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def start_train(self):
        steps = self.cfgs.task.split('+')
        for step in steps:
            if step == 'mobile':
                from models import create_model
            elif step == 'distiller':
                from distiller import create_distiller as create_model
            elif step == 'supernet':
                from supernet import create_supernet as create_model
            else:
                raise NotImplementedError

            print(
                "============================= start train {} ==============================".
                format(step))
            fluid.enable_imperative()
            model = create_model(self.cfgs)
            model.setup()

            _train_dataloader, _ = create_data(self.cfgs)

            epochs = getattr(self.cfgs, '{}_epoch'.format(step))

            for epoch_id in range(epochs):
                for batch_id, data in enumerate(_train_dataloader()):
                    start_time = time.time()
                    model.set_input(data)
                    model.optimize_parameter()
                    batch_time = time.time() - start_time
                    if batch_id % self.cfgs.print_freq == 0:
                        message = 'epoch: %d, batch: %d batch_time: %fs' % (
                            epoch_id, batch_id, batch_time)
                        for k, v in model.get_current_lr().items():
                            message += '%s: %f ' % (k, v)
                        message += '\n'
                        for k, v in model.get_current_loss().items():
                            message += '%s: %.3f ' % (k, v)
                        logging.info(message)

                if epoch_id % self.cfgs.save_freq == 0 or epoch_id == (
                        epochs - 1):
                    model.evaluate_model(epoch_id)
                    model.save_network(epoch_id)
                    if epoch_id == (epochs - 1):
                        model.save_network('last')
            print("=" * 80)


if __name__ == '__main__':
    cfg_instance = configs()
    cfgs = cfg_instance.get_all_config()
    cfg_instance.print_configs(cfgs)
    compression = gan_compression(cfgs)
    compression.start_train()

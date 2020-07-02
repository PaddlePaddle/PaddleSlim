#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import logging

import paddle.fluid as fluid
from dataset.data_loader import create_data
from utils.get_args import configs


class gan_compression:
    def __init__(self, cfgs, **kwargs):
        self.cfgs = cfgs
        use_gpu, use_parallel = self._get_device()

        if not use_gpu:
            place = fluid.CPUPlace()
        else:
            if not use_parallel:
                place = fluid.CUDAPlace(0)
            else:
                place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

        setattr(self.cfgs, 'use_gpu', use_gpu)
        setattr(self.cfgs, 'use_parallel', use_parallel)
        setattr(self.cfgs, 'place', place)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_device(self):
        num = self.cfgs.gpu_num

        use_gpu, use_parallel = False, False
        if num == -1:
            use_gpu = False
        else:
            use_gpu = True
            if num > 1:
                use_parallel = True
        return use_gpu, use_parallel

    def start_train(self):
        steps = self.cfgs.task.split('+')
        for step in steps:
            if step == 'mobile':
                from models import create_model
            elif step == 'distiller':
                from distillers import create_distiller as create_model
            elif step == 'supernet':
                from supernets import create_supernet as create_model
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

                save_model = (not self.cfgs.use_parallel) or (
                    self.cfgs.use_parallel and
                    fluid.dygraph.parallel.Env().local_rank == 0)
                if epoch_id % self.cfgs.save_freq == 0 or epoch_id == (
                        epochs - 1) and save_model:
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

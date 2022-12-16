import os
import sys
import unittest
sys.path.append("../../")
import numpy as np
import paddle
from paddle.io import Dataset
from paddleslim.auto_compression import AutoCompression
paddle.enable_static()


class RandomEvalDataset(Dataset):
    def __init__(self, num_samples, image_shape=[1, 28, 28], class_num=10):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.class_num = class_num

    def __getitem__(self, idx):
        image = np.random.random(self.image_shape).astype('float32')
        return image

    def __len__(self):
        return self.num_samples


class ACTQATWhileOP(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ACTQATWhileOP, self).__init__(*args, **kwargs)
        if not os.path.exists('mnist_while'):
            os.system(
                "wget -q http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
            )
            os.system('tar -xzvf mnist_while.tar.gz')
        self.create_dataloader()
        self.get_config()

    def create_dataloader(self):
        # define a random dataset
        self.eval_dataset = RandomEvalDataset(32)

    def get_config(self):
        self.config = {
            'QuantAware': {},
            'Distillation': {},
            'TrainConfig': {
                'epochs': 1,
                'eval_iter': 100,
                'learning_rate': 5.0e-03,
                'optimizer_builder': {
                    'optimizer': {
                        'type': 'SGD'
                    },
                    "weight_decay": 0.0005,
                }
            }
        }

    def test_demo(self):
        image = paddle.static.data(
            name='x', shape=[-1, 1, 28, 28], dtype='float32')
        train_loader = paddle.io.DataLoader(
            self.eval_dataset, feed_list=[image], batch_size=4)

        ac = AutoCompression(
            model_dir="./mnist_while",
            model_filename="model.pdmodel",
            params_filename="model.pdiparams",
            config=self.config,
            save_dir="qat_while_output",
            train_dataloader=train_loader)
        ac.compress()
        os.system('rm -rf qat_while_output')


if __name__ == '__main__':
    unittest.main()

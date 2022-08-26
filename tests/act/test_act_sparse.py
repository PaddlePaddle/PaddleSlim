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
    def __init__(self, num_samples, image_shape=[3, 398, 224], class_num=10):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.class_num = class_num

    def __getitem__(self, idx):
        image = np.random.random(self.image_shape).astype('float32')
        return image

    def __len__(self):
        return self.num_samples


class ACTSparse(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ACTSparse, self).__init__(*args, **kwargs)
        if not os.path.exists('ppseg_lite_portrait_398x224_with_softmax'):
            os.system(
                "wget -q https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz"
            )
            os.system(
                'tar -xzvf ppseg_lite_portrait_398x224_with_softmax.tar.gz')
        self.create_dataloader()
        self.get_train_config()

    def create_dataloader(self):
        # define a random dataset
        self.eval_dataset = RandomEvalDataset(32)

    def get_train_config(self):
        self.train_config = {
            'TrainConfig': {
                'epochs': 1,
                'eval_iter': 1,
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
            name='x', shape=[-1, 3, 398, 224], dtype='float32')
        train_loader = paddle.io.DataLoader(
            self.eval_dataset, feed_list=[image], batch_size=4)

        ac = AutoCompression(
            model_dir="./ppseg_lite_portrait_398x224_with_softmax",
            model_filename="model.pdmodel",
            params_filename="model.pdiparams",
            input_shapes=[1, 3, 398, 224],
            config=self.train_config,
            save_dir="ppliteseg_output",
            train_dataloader=train_loader,
            deploy_hardware='SD710')
        ac.compress()
        os.system('rm -rf ppliteseg_output')


if __name__ == '__main__':
    unittest.main()

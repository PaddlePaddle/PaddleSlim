import os
import sys
sys.path.append("../../")
import numpy as np
import unittest
import paddle
from paddle.io import Dataset
from paddleslim.common import load_config
from paddleslim.auto_compression.compressor import AutoCompression


class RandomDataset(Dataset):
    def __init__(self, num_samples, sample_shape=[128]):
        self.num_samples = num_samples
        self.sample_shape = sample_shape

    def __getitem__(self, idx):
        input_ids = np.random.random(self.sample_shape).astype('int64')
        token_type_ids = np.random.random(self.sample_shape).astype('int64')
        return input_ids, token_type_ids

    def __len__(self):
        return self.num_samples


class RandomEvalDataset(Dataset):
    def __init__(self, num_samples, sample_shape=[128]):
        self.num_samples = num_samples
        self.sample_shape = sample_shape

    def __getitem__(self, idx):
        input_ids = np.random.random(self.sample_shape).astype('int64')
        token_type_ids = np.random.random(self.sample_shape).astype('int64')
        labels = np.ones(([1])).astype('int64')
        return input_ids, token_type_ids, labels

    def __len__(self):
        return self.num_samples


### select transformer_prune and qat
class NLPAutoCompress(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NLPAutoCompress, self).__init__(*args, **kwargs)
        paddle.enable_static()
        if not os.path.exists('afqmc'):
            os.system(
                'wget -q https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar'
            )
            os.system('tar -xf afqmc.tar')
        self.create_dataset()
        self.get_train_config()

    def create_dataset(self):
        self.fake_dataset = RandomDataset(32)
        self.fake_eval_dataset = RandomEvalDataset(32)

    def get_train_config(self):
        self.train_config = {
            'TrainConfig': {
                'epochs': 1,
                'eval_iter': 1,
                'learning_rate': 2.0e-5,
                'optimizer_builder': {
                    'optimizer': {
                        'type': 'AdamW'
                    },
                    'weight_decay': 0.01
                },
            }
        }

    def test_nlp(self):
        input_ids = paddle.static.data(
            name='input_ids', shape=[-1, -1], dtype='int64')
        token_type_ids = paddle.static.data(
            name='token_type_ids', shape=[-1, -1], dtype='int64')
        labels = paddle.static.data(name='labels', shape=[-1], dtype='int64')
        train_loader = paddle.io.DataLoader(
            self.fake_dataset,
            feed_list=[input_ids, token_type_ids],
            batch_size=32,
            return_list=False)
        eval_loader = paddle.io.DataLoader(
            self.fake_eval_dataset,
            feed_list=[input_ids, token_type_ids, labels],
            batch_size=32,
            return_list=False)

        ac = AutoCompression(
            model_dir='afqmc',
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            config=self.train_config,
            save_dir="nlp_ac_output",
            train_dataloader=train_loader,
            eval_dataloader=eval_loader)
        ac.compress()
        os.system("rm -rf nlp_ac_output")
        os.system("rm -rf afqmc*")


if __name__ == '__main__':
    unittest.main()

import os
import sys
import numpy as np
from tqdm import tqdm
import unittest
sys.path.append("../../")
import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddleslim.auto_compression import AutoCompression
from paddleslim.auto_compression.analysis import analysis_prune
paddle.enable_static()


class ImageNetDataset(DatasetFolder):
    def __init__(self, data_dir, image_size=224, mode='train'):
        super(ImageNetDataset, self).__init__(data_dir)
        self.data_dir = data_dir
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(image_size),
            transforms.Transpose(), normalize
        ])
        self.mode = mode
        train_file_list = os.path.join(data_dir, 'train_list.txt')
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        self.mode = mode
        if mode == 'train':
            with open(train_file_list) as flist:
                full_lines = [line.strip() for line in flist]
                np.random.shuffle(full_lines)
                lines = full_lines
            self.samples = [line.split() for line in lines]
        else:
            with open(val_file_list) as flist:
                lines = [line.strip() for line in flist]
                self.samples = [line.split() for line in lines]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if self.mode == 'train':
            return self.transform(
                Image.open(os.path.join(self.data_dir, img_path)).convert(
                    'RGB'))
        else:
            return self.transform(
                Image.open(os.path.join(self.data_dir, img_path)).convert(
                    'RGB')), np.array([label]).astype('int64')

    def __len__(self):
        return len(self.samples)


def eval_func(program, exe, feed_names, fetch_list, dataloader):
    results = []
    with tqdm(
            total=len(dataloader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, data in enumerate(dataloader):
            image = data[0]['inputs']
            label = data[0]['labels']
            # top1_acc, top5_acc
            if len(feed_names) == 1:
                image = np.array(image)
                label = np.array(label).astype('int64')
                pred = exe.run(program,
                               feed={feed_names[0]: image},
                               fetch_list=fetch_list)
                pred = np.array(pred[0])
                label = np.array(label)
                sort_array = pred.argsort(axis=1)
                top_1_pred = sort_array[:, -1:][:, ::-1]
                top_1 = np.mean(label == top_1_pred)
                top_5_pred = sort_array[:, -5:][:, ::-1]
                acc_num = 0
                for i in range(len(label)):
                    if label[i][0] in top_5_pred[i]:
                        acc_num += 1
                top_5 = float(acc_num) / len(label)
                results.append([top_1, top_5])
            else:
                image = np.array(image)
                label = np.array(label).astype('int64')
                result = exe.run(
                    program,
                    feed={feed_names[0]: image,
                          feed_names[1]: label},
                    fetch_list=fetch_list)
                result = [np.mean(r) for r in result]
                results.append(result)
            t.update()
    result = np.mean(np.array(results), axis=0)
    return result[0]


class ACTChannelPrune(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ACTChannelPrune, self).__init__(*args, **kwargs)
        if not os.path.exists('MobileNetV1_infer'):
            os.system(
                'wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
            )
            os.system('tar -xf MobileNetV1_infer.tar')
        if not os.path.exists('ILSVRC2012_data_demo'):
            os.system(
                'wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz'
            )
            os.system('tar -xf ILSVRC2012_data_demo.tar.gz')

        self.train_dataloader, self.eval_dataloader = self.create_dataloader()

    def create_dataloader(self):
        train_dataset = ImageNetDataset("./ILSVRC2012_data_demo/ILSVRC2012/")
        image = paddle.static.data(
            name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
        label = paddle.static.data(
            name='labels', shape=[None] + [1], dtype='float32')
        train_dataloader = paddle.io.DataLoader(
            train_dataset,
            feed_list=[image],
            batch_size=32,
            shuffle=True,
            num_workers=0,
            return_list=False)

        def eval_reader(data_dir,
                        batch_size,
                        crop_size,
                        resize_size,
                        place=None):
            val_dataset = ImageNetDataset(
                "./ILSVRC2012_data_demo/ILSVRC2012/", mode='val')
            val_loader = paddle.io.DataLoader(
                val_dataset,
                feed_list=[image, label],
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                return_list=False)
            return val_loader

        val_loader = eval_reader(
            './ILSVRC2012_data_demo/ILSVRC2012/',
            batch_size=32,
            crop_size=224,
            resize_size=256)
        return train_dataloader, val_loader

    def get_analysis(self):
        def eval_function(compiled_test_program, exe, test_feed_names,
                          test_fetch_list):
            res = eval_func(compiled_test_program, exe, test_feed_names,
                            test_fetch_list, self.eval_dataloader)
            return res

        ratios = analysis_prune(eval_function, './MobileNetV1_infer',
                                'inference.pdmodel', 'inference.pdiparams',
                                'senti.data', [0.1], 0.05)
        return ratios

    def test_ac_prune_name_is_None(self):
        def eval_function(exe, compiled_test_program, test_feed_names,
                          test_fetch_list):
            res = eval_func(compiled_test_program, exe, test_feed_names,
                            test_fetch_list, self.eval_dataloader)
            return res

        configs = {
            'Distillation': {},
            'ChannelPrune': {
                'pruned_ratio': 0.1
            },
            'TrainConfig': {
                'epochs': 1,
                'eval_iter': 1000,
                'learning_rate': 5.0e-03,
                'optimizer_builder': {
                    'optimizer': {
                        'type': 'SGD'
                    },
                    "weight_decay": 0.0005,
                }
            }
        }

        ac = AutoCompression(
            model_dir='./MobileNetV1_infer',
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            save_dir="prune_output",
            config=configs,
            train_dataloader=self.train_dataloader,
            eval_callback=eval_function)  # eval_function to verify accuracy
        ac.compress()
        os.system('rm -rf prune_output')

    def test_ac_prune(self):
        ratios = self.get_analysis()

        def eval_function(exe, compiled_test_program, test_feed_names,
                          test_fetch_list):
            res = eval_func(compiled_test_program, exe, test_feed_names,
                            test_fetch_list, self.eval_dataloader)
            return res

        configs = {
            'Distillation': {},
            'TrainConfig': {
                'epochs': 1,
                'eval_iter': 1000,
                'learning_rate': 5.0e-03,
                'optimizer_builder': {
                    'optimizer': {
                        'type': 'SGD'
                    },
                    "weight_decay": 0.0005,
                }
            }
        }
        configs.update({
            'ChannelPrune': {
                'prune_params_name': list(ratios.keys())
            }
        })
        configs['ChannelPrune'].update({'pruned_ratio': list(ratios.values())})

        ac = AutoCompression(
            model_dir='./MobileNetV1_infer',
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            save_dir="prune_output",
            config=configs,
            train_dataloader=self.train_dataloader,
            eval_callback=eval_function)  # eval_function to verify accuracy
        ac.compress()
        os.system('rm -rf prune_output')

    def test_ac_sparse(self):
        def eval_function(exe, compiled_test_program, test_feed_names,
                          test_fetch_list):
            res = eval_func(compiled_test_program, exe, test_feed_names,
                            test_fetch_list, self.eval_dataloader)
            return res

        configs = {
            'Distillation': {},
            'ASPPrune': {},
            'TrainConfig': {
                'epochs': 1,
                'eval_iter': 1000,
                'learning_rate': 5.0e-03,
                'optimizer_builder': {
                    'optimizer': {
                        'type': 'SGD'
                    },
                    "weight_decay": 0.0005,
                }
            }
        }

        ac = AutoCompression(
            model_dir='./MobileNetV1_infer',
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            save_dir="asp_output",
            config=configs,
            train_dataloader=self.train_dataloader,
            eval_callback=eval_function)  # eval_function to verify accuracy
        ac.compress()
        os.system('rm -rf asp_output')


if __name__ == '__main__':
    unittest.main()

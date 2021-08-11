import os
import numpy as np
from PIL import Image
from paddle.io import Dataset
from paddle.vision.transforms import transforms


class ImageNetDataset(Dataset):
    def __init__(self,
                 data_dir,
                 mode='train',
                 image_size=224,
                 resize_short_size=256):
        super(ImageNetDataset, self).__init__()
        train_file_list = os.path.join(data_dir, 'train_list.txt')
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        test_file_list = os.path.join(data_dir, 'test_list.txt')
        self.data_dir = data_dir
        self.mode = mode

        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(), transforms.Transpose(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize_short_size),
                transforms.CenterCrop(image_size), transforms.Transpose(),
                normalize
            ])

        if mode == 'train':
            with open(train_file_list) as flist:
                full_lines = [line.strip() for line in flist]
                np.random.shuffle(full_lines)
                if os.getenv('PADDLE_TRAINING_ROLE'):
                    # distributed mode if the env var `PADDLE_TRAINING_ROLE` exits
                    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                    trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
                    per_node_lines = len(full_lines) // trainer_count
                    lines = full_lines[trainer_id * per_node_lines:(
                        trainer_id + 1) * per_node_lines]
                    print(
                        "read images from %d, length: %d, lines length: %d, total: %d"
                        % (trainer_id * per_node_lines, per_node_lines,
                           len(lines), len(full_lines)))
                else:
                    lines = full_lines
            self.data = [line.split() for line in lines]
        else:
            with open(val_file_list) as flist:
                lines = [line.strip() for line in flist]
                self.data = [line.split() for line in lines]

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img_path = os.path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        label = np.array([label]).astype(np.int64)
        return self.transform(img), label

    def __len__(self):
        return len(self.data)

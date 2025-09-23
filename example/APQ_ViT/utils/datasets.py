import paddle
import paddle.io as io
import numpy as np
import copy
import paddle
import paddle.vision.transforms as T
import paddle.vision.datasets as datasets
import os

class LoaderGenerator():
    """
    A class for managing dataset loaders in PaddlePaddle.
    """
    def __init__(self,
                 root,
                 dataset_name,
                 train_batch_size=1,
                 test_batch_size=1,
                 num_workers=0,
                 kwargs={}):
        self.root = root
        self.dataset_name = str.lower(dataset_name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.items = []
        self._train_set = None
        self._test_set = None
        self._calib_set = None
        self.train_transform = None
        self.test_transform = None
        self.train_loader_kwargs = {
            'num_workers': self.num_workers,
            'drop_last': kwargs.get('drop_last', False)
        }
        self.test_loader_kwargs = self.train_loader_kwargs.copy()
        self.load()

    @property
    def train_set(self):
        if self._train_set is None:
            raise ValueError("Training set has not been loaded yet.")
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            raise ValueError("Test set has not been loaded yet.")
        return self._test_set

    def load(self):
        """
        Implement this method to load your dataset into self._train_set, self._test_set.
        """
        pass

    def train_loader(self):
        assert self.train_set is not None
        return io.DataLoader(self.train_set,
                             batch_size=self.train_batch_size,
                             shuffle=True,
                             **self.train_loader_kwargs)

    def test_loader(self, shuffle=False, batch_size=None):
        assert self.test_set is not None
        if batch_size is None:
            batch_size = self.test_batch_size
        return io.DataLoader(self.test_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             **self.test_loader_kwargs)

    def val_loader(self):
        assert self.val_set is not None
        return io.DataLoader(self.val_set,
                             batch_size=self.test_batch_size,
                             shuffle=False,
                             **self.test_loader_kwargs)

    def trainval_loader(self):
        assert self.trainval_set is not None
        return io.DataLoader(self.trainval_set,
                             batch_size=self.train_batch_size,
                             shuffle=True,
                             **self.train_loader_kwargs)

    def calib_loader(self, num=1024, seed=3):
        if self._calib_set is None:
            np.random.seed(seed)
            # Randomly sample indices for calibration
            inds = np.random.permutation(len(self.train_set))[:num]
            self._calib_set = io.Subset(
                copy.deepcopy(self.train_set), inds)
            self._calib_set.dataset.transform = self.test_transform
        return io.DataLoader(self._calib_set,
                             batch_size=num,
                             shuffle=False,
                             **self.train_loader_kwargs)
    

class ImageNetLoaderGenerator(LoaderGenerator):
    def load(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.train_transform = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

        self.test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = datasets.ImageNet(
                mode='train',  # 'train' for training set
                transform=self.train_transform,
                data_dir=os.path.join(self.root, 'train')  # path to the train data
            )
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = datasets.ImageNet(
                mode='val',  # 'val' for validation set
                transform=self.test_transform,
                data_dir=os.path.join(self.root, 'val')  # path to the validation data
            )
        return self._test_set
    

class ViTImageNetLoaderGenerator(ImageNetLoaderGenerator):

    def __init__(self,
                 root,
                 dataset_name,
                 train_batch_size,
                 test_batch_size,
                 num_workers,
                 kwargs={}):
        kwargs.update({"pin_memory": False})
        super().__init__(root,
                         dataset_name,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         num_workers=num_workers,
                         kwargs=kwargs)

    def load(self):
        model = self.kwargs.get("model", None)
        assert model is not None, f"No model in ViTImageNetLoaderGenerator!"

        # Resolve data configuration based on the model
        config = self.resolve_data_config(model)

        # Create transforms for training and testing
        self.train_transform = self.create_transform(config, is_training=True)
        self.test_transform = self.create_transform(config, is_training=False)


    def resolve_data_config(self, model):
        """
        Resolve configuration for data preprocessing based on the model.
        For simplicity, we assume it's a ViT model and return ImageNet stats.
        """
        
        return {
                'input_size': 224,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
                }
    

    def create_transform(self, config, is_training=False):
        """
        Create necessary transforms based on training or testing phase.
        """
        transforms_list = []

        # Add RandomResizedCrop and RandomHorizontalFlip for training
        if is_training:
            transforms_list.append(T.RandomResizedCrop(config['input_size']))
            transforms_list.append(T.RandomHorizontalFlip())

        # Normalize with ImageNet mean and std
        transforms_list.append(T.Normalize(mean=config['mean'], std=config['std']))

        # Convert image to Tensor
        transforms_list.append(T.ToTensor())

        return T.Compose(transforms_list)
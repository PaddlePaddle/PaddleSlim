"""
data loder for loading data
"""
import os
import math
import paddle
from paddle.vision.transforms import transforms
import numpy as np
from PIL import Image
import struct

__all__ = ["DataLoader", "PartDataLoader"]


class ImageLoader():
	def __init__(self, dataset_dir, transform=None, target_transform=None):
		class_list = os.listdir(dataset_dir)
		datasets = []
		for cla in class_list:
			cla_path = os.path.join(dataset_dir, cla)
			files = os.listdir(cla_path)
			for file_name in files:
				file_path = os.path.join(cla_path, file_name)
				if os.path.isfile(file_path):
					# datasets.append((file_path, tuple([float(v) for v in int(cla)])))
					datasets.append((file_path, [float(cla)]))
					# print(datasets)
					# assert False
		
		self.dataset_dir = dataset_dir
		self.datasets = datasets
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		frames = []
		
		file_path, label = self.datasets[index]
		noise = paddle.load(file_path)
		return noise, paddle.to_tensor(label)
	
	def __len__(self):
		return len(self.datasets)


class DataLoader(object):
	"""
	data loader for CV data sets
	"""
	
	def __init__(self, dataset, batch_size, n_threads=4,
	             ten_crop=False, data_path='/home/dataset/', logger=None):
		"""
		create data loader for specific data set
		:params n_treads: number of threads to load data, default: 4
		:params ten_crop: use ten crop for testing, default: False
		:params data_path: path to data set, default: /home/dataset/
		"""
		self.dataset = dataset
		self.batch_size = batch_size
		self.n_threads = n_threads
		self.ten_crop = ten_crop
		self.data_path = data_path
		self.logger = logger
		self.dataset_root = data_path
		
		self.logger.info("|===>Creating data loader for " + self.dataset)
		
		if self.dataset in ["cifar100"]:
			self.train_loader, self.test_loader = self.cifar(
				dataset=self.dataset)
		
		elif self.dataset in ["imagenet"]:
			self.train_loader, self.test_loader = self.imagenet(
				dataset=self.dataset)
		else:
			assert False, "invalid data set"
	
	def getloader(self):
		"""
		get train_loader and test_loader
		"""
		return self.train_loader, self.test_loader

	def imagenet(self, dataset="imagenet", data_path: str = '/home/xingyu-zheng/laboratory/data/imagenet', input_size: int = 224, batch_size: int = 16, workers: int = 2, dist_sample: bool = False):
		print('==> Using Pytorch Dataset')

		traindir = os.path.join(data_path, 'train')
		valdir = os.path.join(data_path, 'val')
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])

		import paddle.io as io
		from paddle.vision.datasets import DatasetFolder
		#torchvision.set_image_backend('accimage')
		train_dataset = DatasetFolder(
			traindir,
			transform=transforms.Compose([
				transforms.RandomResizedCrop(input_size),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]))
		val_dataset = DatasetFolder(
			valdir,
			transform=transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(input_size),
				transforms.ToTensor(),
				normalize,
			]))

		if dist_sample:
			train_sampler = io.DistributedBatchSampler(train_dataset, batch_size=1)
			val_sampler = io.DistributedBatchSampler(val_dataset, batch_size=1)
		else:
			train_sampler = None
			val_sampler = None

		train_loader = io.DataLoader(
			train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
			num_workers=workers, batch_sampler=train_sampler)
		val_loader = io.DataLoader(
			val_dataset,batch_size=batch_size, shuffle=False,
			num_workers=workers, batch_sampler=val_sampler)
		return train_loader, val_loader

	def cifar(self, dataset="cifar100"):
		"""
		dataset: cifar
		"""
		if dataset == "cifar10":
			norm_mean = [0.49139968, 0.48215827, 0.44653124]
			norm_std = [0.24703233, 0.24348505, 0.26158768]
		elif dataset == "cifar100":
			norm_mean = [0.50705882, 0.48666667, 0.44078431]
			norm_std = [0.26745098, 0.25568627, 0.27607843]
		
		else:
			assert False, "Invalid cifar dataset"

		test_data_root = self.dataset_root

		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(norm_mean, norm_std)])

		if self.dataset == "cifar10":
			test_dataset = dsets.CIFAR10(root=test_data_root,
			                             train=False,
			                             transform=test_transform)
		elif self.dataset == "cifar100":
			test_dataset = dsets.CIFAR100(root=test_data_root,
			                              train=False,
			                              transform=test_transform,
			                              download=True)
		else:
			assert False, "invalid data set"

		test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												  batch_size=200,
												  shuffle=False,
												  pin_memory=True,
												  num_workers=self.n_threads)
		return None, test_loader
	


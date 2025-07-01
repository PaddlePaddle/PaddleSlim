import os
import shutil

from pyhocon import ConfigFactory

from utils.opt_static import NetOption


class Option(NetOption):
	def __init__(self, conf_path):
		super(Option, self).__init__()
		self.conf = ConfigFactory.parse_file(conf_path)
		#  ------------ General options ----------------------------------------
		self.save_path = self.conf['save_path']
		self.dataPath = self.conf['dataPath']  # path for loading data set
		self.dataset = self.conf['dataset']  # options: imagenet | cifar100
		self.nGPU = self.conf['nGPU']  # number of GPUs to use by default
		self.GPU = self.conf['GPU']  # default gpu to use, options: range(nGPU)
		self.visible_devices = self.conf['visible_devices']
		
		# ------------- Data options -------------------------------------------
		self.nThreads = self.conf['nThreads']  # number of data loader threads
		
		# ---------- Optimization options --------------------------------------
		self.nEpochs = self.conf['nEpochs']  # number of total epochs to train
		self.batchSize = self.conf['batchSize']  # mini-batch size
		self.momentum = self.conf['momentum']  # momentum
		self.weightDecay = float(self.conf['weightDecay'])  # weight decay
		self.opt_type = self.conf['opt_type']
		self.warmup_epochs = self.conf['warmup_epochs']  # number of epochs for warmup

		self.lr_S = self.conf['lr_S']  # initial learning rate
		self.lrPolicy_S = self.conf['lrPolicy_S']  # options: multi_step | linear | exp | const | step
		self.step_S = self.conf['step_S']  # step for linear or exp learning rate policy
		self.decayRate_S = self.conf['decayRate_S']  # lr decay rate
		
		# ---------- Model options ---------------------------------------------
		self.experimentID = self.conf['experimentID']
		self.nClasses = self.conf['nClasses']  # number of classes in the dataset
		
		# ---------- Quantization options ---------------------------------------------
		self.qw = self.conf['qw']
		self.qa = self.conf['qa']
		
		# ----------KD options ---------------------------------------------
		self.temperature = self.conf['temperature']
		self.alpha = self.conf['alpha']
		
		# ----------Generator options ---------------------------------------------
		self.latent_dim = self.conf['latent_dim']
		self.img_size = self.conf['img_size']
		self.channels = self.conf['channels']

		self.lr_G = self.conf['lr_G']
		self.lrPolicy_G = self.conf['lrPolicy_G']  # options: multi_step | linear | exp | const | step
		self.step_G = self.conf['step_G']  # step for linear or exp learning rate policy
		self.decayRate_G = self.conf['decayRate_G']  # lr decay rate

		self.b1 = self.conf['b1']
		self.b2 = self.conf['b2']

		
	def set_save_path(self):
		self.save_path = self.save_path + "log_{}_bs{:d}_lr{:.4f}_TELCNN_baseline_opt{}_qw{:d}_qa{:d}_epoch{}_{}/".format(
			self.dataset, self.batchSize, self.lr, self.opt_type, self.qw, self.qa,
			self.nEpochs, self.experimentID)
		
		if os.path.exists(self.save_path):
			print("{} file exist!".format(self.save_path))
			action = input("Select Action: d (delete) / q (quit):").lower().strip()
			act = action
			if act == 'd':
				shutil.rmtree(self.save_path)
			else:
				raise OSError("Directory {} exits!".format(self.save_path))
		
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
	
	def paramscheck(self, logger):
		logger.info("|===>The used PyTorch version is {}".format(
				self.torch_version))
		
		if self.dataset in ["cifar10", "mnist"]:
			self.nClasses = 10
		elif self.dataset == "cifar100":
			self.nClasses = 100
		elif self.dataset == "imagenet" or "thi_imgnet":
			self.nClasses = 1000
		elif self.dataset == "imagenet100":
			self.nClasses = 100
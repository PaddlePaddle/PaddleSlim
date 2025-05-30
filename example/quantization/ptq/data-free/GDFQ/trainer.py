"""
basic trainer
"""
import time

import paddle.nn as nn
import paddle.nn.functional as F
import utils as utils
import numpy as np
import paddle

__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model: nn.Layer, model_teacher: nn.Layer, generator: nn.Layer, lr_master_S, lr_master_G,
	             train_loader, test_loader, settings, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		
		self.model = utils.data_parallel(
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.generator = utils.data_parallel(
			generator, self.settings.nGPU, self.settings.GPU)

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		self.criterion = nn.CrossEntropyLoss()
		self.bce_logits = nn.BCEWithLogitsLoss()
		self.MSE_loss = nn.MSELoss()
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type
		from paddle.optimizer import Momentum, RMSProp, Adam, AdamW
		if opt_type == "SGD":
			self.optimizer_S = Momentum(
				parameters=self.model.parameters(),
				learning_rate=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				# nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = RMSProp(
				parameters=self.model.parameters(),
				learning_rate=self.lr_master_S.lr,
				epsilon=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				# alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = Adam(
				parameters=self.model.parameters(),
				learning_rate=self.lr_master_S.lr,
				epsilon=1e-5,
				weight_decay=self.settings.weightDecay
			)
		elif opt_type == "AdamW":
			self.optimizer_S = AdamW(parameters=self.model.parameters())
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)\

		# self.optimizer_G = Adam(parameters=self.generator.parameters(), learning_rate=self.settings.lr_G,
		# 						beta1=self.settings.b1, beta2=self.settings.b2)
		self.optimizer_G = AdamW(parameters=self.generator.parameters())

		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []

		self.fix_G = False
	
	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		self.optimizer_S.set_lr(lr_S)
		self.optimizer_G.set_lr(lr_G)
	
	def loss_fn_kd(self, output, labels, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""

		criterion_d = nn.CrossEntropyLoss()
		kdloss = nn.KLDivLoss()

		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, axis=1)
		b = F.softmax(teacher_outputs / T, axis=1)
		c = (alpha * T * T)
		d = criterion_d(output, labels)

		KD_loss = kdloss(a,b)*c + d
		return KD_loss
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize
		output = self.model(images)
		if labels is not None:
			loss = self.loss_fn_kd(output, labels, teacher_outputs)
			# loss = paddle.zeros(1)
			return output, loss
		else:
			return output, None
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.clear_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.clear_grad()
		loss_S.backward()
		self.optimizer_S.step()

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer_G.clear_grad()
		self.optimizer_S.clear_grad()
		loss.backward()
		self.optimizer_G.step()
		self.optimizer_S.step()

	def hook_fn_forward(self,module, input: paddle.Tensor, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)
		# import pdb; pdb.set_trace()

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module._mean)
		self.teacher_running_var.append(module._variance)

	def hook_fn_forward_saveBN(self,module, input, output):
		self.save_BN_mean.append(module._mean.cpu())
		self.save_BN_var.append(module._variance.cpu())
	
	def train(self, epoch):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		iters = 200
		self.update_lr(epoch)

		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time
		
		if epoch==0:
			for m in self.model_teacher.sublayers():
				if isinstance(m, nn.BatchNorm2D):
					m.register_forward_post_hook(self.hook_fn_forward)
		
		for i in range(iters):
			# import pdb; pdb.set_trace()
			# print(i)
			start_time = time.time()
			data_time = start_time - end_time

			z = paddle.randn([self.settings.batchSize, self.settings.latent_dim]).cuda() # todo: Variable

			# Get labels ranging from 0 to n_classes for n rows
			labels = paddle.randint(0, self.settings.nClasses, (self.settings.batchSize,)).cuda() # todo: Variable
			# z = z.contiguous()
			# labels = labels.contiguous()
			images = self.generator(z, labels)
		
			self.mean_list.clear()
			self.var_list.clear()
			# import pdb; pdb.set_trace()
			output_teacher_batch = self.model_teacher(images)

			# One hot loss
			loss_one_hot = self.criterion(output_teacher_batch, labels)

			# BN statistic loss
			BNS_loss = paddle.zeros([1]).cuda()

			for num in range(len(self.mean_list)):
				# import pdb; pdb.set_trace()
				# self.var_list[num]-self.teacher_running_var[num]
				BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
					self.var_list[num], self.teacher_running_var[num])

			BNS_loss = BNS_loss / len(self.mean_list)

			# loss of Generator
			loss_G = loss_one_hot + 0.1 * BNS_loss

			self.backward_G(loss_G)

			output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), labels) # OOM here
			
			if epoch>= self.settings.warmup_epochs:
				self.backward_S(loss_S)


			single_error, single_loss, single5_error = utils.compute_singlecrop(
				outputs=output, labels=labels,
				loss=loss_S, top5_flag=True, mean_flag=True)
			
			top1_error.update(single_error, images.shape[0])
			top1_loss.update(single_loss, images.shape[0])
			top5_error.update(single5_error, images.shape[0])
			
			end_time = time.time()
			
			gt = labels.cpu().numpy()
			d_acc = np.mean(np.argmax(output_teacher_batch.cpu().numpy(), axis=1) == gt)

			fp_acc.update(d_acc)

		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f] [S loss: %f] "
			% (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(),
			 loss_S.item())
		)

		self.scalar_info['accuracy every epoch'] = 100 * d_acc
		self.scalar_info['G loss every epoch'] = loss_G
		self.scalar_info['One-hot loss every epoch'] = loss_one_hot
		self.scalar_info['S loss every epoch'] = loss_S

		self.scalar_info['training_top1error'] = top1_error.avg
		self.scalar_info['training_top5error'] = top5_error.avg
		self.scalar_info['training_loss'] = top1_loss.avg
		
		if self.tensorboard_logger is not None:
			for tag, value in list(self.scalar_info.items()):
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}

		return top1_error.avg, top1_loss.avg, top5_error.avg
	
	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with paddle.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output = self.model(images)

				loss = paddle.ones(1)
				self.mean_list.clear()
				self.var_list.clear()

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.shape[0])
				top1_loss.update(single_loss, images.shape[0])
				top5_error.update(single5_error, images.shape[0])
				
				end_time = time.time()
		
		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		
		self.scalar_info['testing_top1error'] = top1_error.avg
		self.scalar_info['testing_top5error'] = top5_error.avg
		self.scalar_info['testing_loss'] = top1_loss.avg
		if self.tensorboard_logger is not None:
			for tag, value in self.scalar_info.items():
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}
		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg

	def test_teacher(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with paddle.no_grad():
			# import pdb; pdb.set_trace()
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.cuda()
				if self.settings.tenCrop:
					image_size = images.shape()
					images = images.reshape([image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3]])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.cuda()
						img_var = img # todo: Variable
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = paddle.concat((output, temp_output.numpy()))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.cuda()

					output = self.model_teacher(images)

					loss = paddle.ones(1)
					self.mean_list.clear()
					self.var_list.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.shape[0])
				top1_loss.update(single_loss, images.shape[0])
				top5_error.update(single5_error, images.shape[0])

				end_time = time.time()
				iter_time = end_time - start_time

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import copy
from tensorboardX import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune, sws_prune_l2, sws_prune_0
from utils_misc import trueAfterN
import pickle


###Training and testing NN
def test_accuracy(data, labels, model, loss_type='CE'):
	outputs = model(data)
	if loss_type == 'CE':
		loss = nn.CrossEntropyLoss()(outputs, labels).data[0]
	else:
		loss = nn.MSELoss(outputs, labels).data[0]
	_, predicted = torch.max(outputs.data, 1)
	correct = (predicted == labels.data).sum()
	accuracy = 100.0 * correct/len(labels)
	return accuracy, loss
	
def train_epoch(model, optimizer, criterion, train_loader):
	loss_total = 0
	for i, (images, labels) in enumerate(train_loader):
		#if(use_cuda):
		images=images.cuda()
		labels=labels.cuda()
		images = Variable(images)
		labels = Variable(labels)
		# Clear gradients w.r.t. parameters
		optimizer.zero_grad()
		# Forward pass to get output/logits
		outputs = nn.Softmax(dim=1)(model(images))
		# Calculate Loss: softmax --> cross entropy loss
		loss = criterion(outputs, labels)
		loss_total += float(loss[0])
		# Getting gradients w.r.t. parameters
		loss.backward()
		# Updating parameters
		optimizer.step()
	return model, loss_total

def train_epoch_l2(model, optimizer, criterion, train_loader, w = 0.0001):
	loss_total = 0
	complexity_loss = Variable(torch.Tensor([0])).cuda()
	for i, (images, labels) in enumerate(train_loader):
		#if(use_cuda):
		images=images.cuda()
		labels=labels.cuda()
		images = Variable(images)
		labels = Variable(labels)
		# Clear gradients w.r.t. parameters
		optimizer.zero_grad()
		for layer in model.state_dict():
			complexity_loss += torch.pow(model.state_dict()[layer], 2).sum()
		# Forward pass to get output/logits
		outputs = nn.Softmax(dim=1)(model(images))
		# Calculate Loss: softmax --> cross entropy loss
		loss_acc = criterion(outputs, labels)
		#loss_total += float(loss_acc[0])
		
		loss = loss_acc + w * complexity_loss
		# Getting gradients w.r.t. parameters
		loss.backward()
		# Updating parameters
		optimizer.step()
	return model, loss_total

###

			
def get_weight_penalty(model):
	layer_list = [x.replace(".weight","") for x in model.state_dict().keys() if 'weight' in x]
	wp=0
	for layer in layer_list:
		wp += np.sqrt( ( model.state_dict()[layer + ".weight"].pow(2).sum() + model.state_dict()[layer + ".bias"].pow(2).sum() ) )
	return wp 

###
class model_prune():
	def __init__(self, state_dict):
		self.state_dict = copy.deepcopy(state_dict)
		self.std = {}
		self.mean = {}
		self.num_weights = {}
		self.percentile_limits = {}
		self.prune_list = [x for x in self.state_dict.keys() if 'weight' in x]
		self.num_pruned = 0
		for layer in self.state_dict:
			self.std[layer] = self.state_dict[layer].std()
			self.mean[layer] = self.state_dict[layer].mean()
			self.num_weights[layer] = 1
			for dim in self.state_dict[layer].size():
				self.num_weights[layer] *= dim
			weight_np = np.abs((self.state_dict[layer].clone().cpu().numpy())).reshape(-1)
			self.percentile_limits[layer] = np.percentile(weight_np, range(0,101))
		self.total_weights = sum(self.num_weights.values())
			
	def percentile_prune(self, percentile):
		new_state_dict = copy.deepcopy(self.state_dict)
		self.num_pruned = 0
		for layer in self.prune_list:
			zero_idx = new_state_dict[layer].abs()<self.percentile_limits[layer][percentile]
			new_state_dict[layer][zero_idx] = 0
			self.num_pruned += zero_idx.sum()
		return new_state_dict
	
	def deviation_prune(self, deviation):
		new_state_dict = copy.deepcopy(self.state_dict)
		self.num_pruned = 0
		for layer in self.prune_list:
			zero_idx = (new_state_dict[layer] - self.mean[layer]).abs() < self.std[layer] * deviation
			new_state_dict[layer][zero_idx] = self.mean[layer]
			self.num_pruned += zero_idx.sum()
		return new_state_dict   
	
###
def retrain_sws_epoch(model, gmp, optimizer, train_loader, tau, temp = 1.0, loss_type='MSESNT'):#, optimizer_gmp, optimizer_gmp2
	"""
	train model
	model: neural network model
	optimizer: optimization algorithm/configuration
	criterion: loss function
	train_loader: training dataset dataloader
	temp: KD temperature - 0 for logits
	Loss: CE, 
	"""
	for i, (images, targets) in enumerate(train_loader):
		#if(use_cuda):
		images=images.cuda()
		targets=targets.cuda()
		images = Variable(images)
		targets = Variable(targets)
		optimizer.zero_grad()
		# Forward pass to get output/logits
		forward = model(images)
		if (loss_type == 'CEST'):
			outputs = nn.LogSoftmax(dim=1)(forward/temp)
			loss_acc = -torch.mean(torch.sum(targets * outputs, dim=1)) * temp

		if (loss_type == 'CESNT'):
			outputs = nn.Softmax(dim=1)(forward)
			loss_acc = nn.CrossEntropyLoss()(outputs, targets)

		if (loss_type == 'CESH'):
			outputs = nn.LogSoftmax(dim=1)((nn.ReLU()(forward))/temp)
			loss_acc = -torch.mean(torch.sum(targets * outputs, dim=1)) * temp

		if (loss_type == 'MSEST'):
			outputs = nn.Softmax(dim=1)(forward/temp)
			loss_acc = nn.MSELoss()(outputs, targets) * temp

		if (loss_type == 'MSESNT'):
			outputs = nn.Softmax(dim=1)(forward)
			loss_acc = nn.MSELoss()(outputs, targets.float())

		if (loss_type == 'MSEHA'):
			outputs = nn.ReLU()(forward)
			loss_acc = nn.MSELoss()(outputs, targets)
			
		if (loss_type == 'MSEHNA' or loss_type == 'MSEL'):
			outputs = forward
			loss_acc = nn.MSELoss()(outputs, targets)
			
		#Loss = CE | MSE , OP type = S | H | L, Temp = T | NT
		#CE - MSE
		#Logits - Temp - Hidden
		#CE Temp - div by temp - softmax - CE loss - mult by temp
		#CE No-Temp - softmax - CE loss
		#CE Hidden (exp) - div by temp - ReLU + softmax - CE Loss mult by temp (experimental)
		#MSE Temp - div by temp - softmax - MSE Loss - mult by temp
		#MSE No-Temp - no temp - softmax - MSE Loss - mult by temp
		#MSE Hidden Activation - no temp - ReLU - MSE Loss - no mult by temp
		#MSE Logit / Hidden No Act - no temp - MSE Loss - no mult by temp
		loss = loss_acc + tau * gmp.call()
		loss.backward()
		# Updating parameters
		optimizer.step()
	return model, loss
	
def train_epoch(model, optimizer, criterion, train_loader):
	for i, (images, labels) in enumerate(train_loader):
		#if(use_cuda):
		images=images.cuda()
		labels=labels.cuda()
		images = Variable(images)
		labels = Variable(labels)
		# Clear gradients w.r.t. parameters
		optimizer.zero_grad()
		# Forward pass to get output/logits
		outputs = model(images)
		# Calculate Loss: softmax --> cross entropy loss
		loss = criterion(outputs, labels)
		# Getting gradients w.r.t. parameters
		loss.backward()
		# Updating parameters
		optimizer.step()
	return model, loss
	
###
def layer_accuracy(model_retrain, gmp, model_orig, data, labels):
	model_acc = copy.deepcopy(model_orig)
	org_acc = (test_accuracy(data, labels, model_orig))

	weight_loader = copy.deepcopy(model_orig.state_dict())
	for layer in model_retrain.state_dict():
		weight_loader[layer] = model_retrain.state_dict()[layer]
	model_acc.load_state_dict(weight_loader)
	retrain_acc = (test_accuracy(data, labels, model_acc))
	model_acc.load_state_dict(model_orig.state_dict())

	model_prune = copy.deepcopy(model_retrain)
	model_prune.load_state_dict(sws_prune_l2(model_prune, gmp))

	weight_loader = copy.deepcopy(model_orig.state_dict())
	for layer in model_prune.state_dict():
		weight_loader[layer] = model_prune.state_dict()[layer]
	model_acc.load_state_dict(weight_loader)
	prune_acc = (test_accuracy(data, labels, model_acc))
	model_acc.load_state_dict(model_orig.state_dict())

	model_prune = copy.deepcopy(model_retrain)
	model_prune.load_state_dict(sws_prune_0(model_prune, gmp))

	sp_zeroes = 0
	sp_elem = 0
	for layer in model_prune.state_dict():
		sp_zeroes += float((model_prune.state_dict()[layer].view(-1) == 0).sum())
		sp_elem += float(model_prune.state_dict()[layer].view(-1).numel())
	sp = sp_zeroes/sp_elem * 100.0

	weight_loader = copy.deepcopy(model_orig.state_dict())
	for layer in model_prune.state_dict():
		weight_loader[layer] = model_prune.state_dict()[layer]
	sp = sp_zeroes/sp_elem * 100.0
	model_acc.load_state_dict(weight_loader)
	prune_0_acc = (test_accuracy(data, labels, model_acc))
	model_acc.load_state_dict(model_orig.state_dict())


	
	print ("Original: {:.2f}% - Retrain: {:.2f}% - Prune: {:.2f}% - Quantize: {:.2f}% - Sparsity: {:.2f}%".format(org_acc[0], retrain_acc[0], prune_0_acc[0], prune_acc[0], sp))
	return retrain_acc[0], prune_0_acc[0], prune_acc[0], sp
	
def sws_replace(model_orig, conv1, conv2, fc1, fc2):
	new_model = copy.deepcopy(model_orig)
	new_dict = new_model.state_dict()
	for layer in conv1:
		new_dict[layer] = conv1[layer]
	for layer in conv2:
		new_dict[layer] = conv2[layer]
	for layer in fc1:
		new_dict[layer] = fc1[layer]
	for layer in fc2:
		new_dict[layer] = fc2[layer]
	new_model.load_state_dict(new_dict)
	return new_model
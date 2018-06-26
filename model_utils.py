import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
from tensorboardX import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt
from sws_utils import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune
from helpers import trueAfterN
import pickle


###Training and testing NN
def test_accuracy(data, labels, model):
	outputs = model(data)
	loss = nn.CrossEntropyLoss()(outputs, labels).data[0]
	_, predicted = torch.max(outputs.data, 1)
	correct = (predicted == labels.data).sum()
	accuracy = 100.0 * correct/len(labels)
	return accuracy, loss
	
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
def retrain_sws_epoch(model, gmp, optimizer, optimizer_gmp, optimizer_gmp2, criterion, train_loader, tau):
	"""
	train model
	
	model: neural network model
	optimizer: optimization algorithm/configuration
	criterion: loss function
	train_loader: training dataset dataloader
	"""
	model.train()
	for i, (images, labels) in enumerate(train_loader):
		#if(use_cuda):
		images=images.cuda()
		labels=labels.cuda()
		images = Variable(images)
		labels = Variable(labels)
		# Clear gradients w.r.t. parameters
		optimizer.zero_grad()
		optimizer_gmp.zero_grad()
		optimizer_gmp2.zero_grad()
		# Forward pass to get output/logits
		outputs = model(images)
		# Calculate Loss: softmax --> cross entropy loss
		#loss = criterion(outputs, labels) + 0.001 * ( (model.fc1.weight - 0.05).norm() + (model.fc2.weight - 0.05).norm() + (model.fc3.weight - 0.05).norm() + (model.fc1.weight + 0.05).norm() + (model.fc2.weight + 0.05).norm() + (model.fc3.weight + 0.05).norm())
		loss = criterion(outputs, labels)
		#print (criterion(outputs, labels))
		#print (gmp.call())
		# Getting gradients w.r.t. parameters
		gmp_loss = tau * gmp.call()
		loss.backward()
		gmp_loss.backward()
		# Updating parameters
		optimizer.step()
		optimizer_gmp.step()
		optimizer_gmp2.step()
	return model, criterion(outputs, labels)
	
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
	model_prune.load_state_dict(sws_prune(model_prune, gmp))

	weight_loader = copy.deepcopy(model_orig.state_dict())
	for layer in model_prune.state_dict():
		weight_loader[layer] = model_prune.state_dict()[layer]
	model_acc.load_state_dict(weight_loader)
	prune_acc = (test_accuracy(data, labels, model_acc))
	model_acc.load_state_dict(model_orig.state_dict())
	
	print ("Original: {:.2f}% - Retrain: {:.2f}% - Prune: {:.2f}%".format(org_acc[0], retrain_acc[0], prune_acc[0]))
	
def retrain_layer(model_retrain, model_orig, data_loader, test_data_full, test_labels_full, alpha, beta, tau, mixtures, model_dir):
	
	weight_loader = model_retrain.state_dict()
	for layer in model_retrain.state_dict():
		weight_loader[layer] = model_orig.state_dict()[layer]
	model_retrain.load_state_dict(weight_loader)

	gmp = GaussianMixturePrior(mixtures, [x for x in model_retrain.parameters()], 0.99, ab = (alpha, beta))

	sws_param1 = [gmp.means]
	sws_param2 = [gmp.gammas, gmp.rhos]

	print ("Model Name: {}".format(model_retrain.name))
	criterion = nn.MSELoss()
	optimizer_kd = torch.optim.SGD(model_retrain.parameters(), lr=1e-3, weight_decay = 0)
	optimizer_gmp1 = torch.optim.SGD(sws_param1, lr=1e-4)
	optimizer_gmp2 = torch.optim.SGD(sws_param2, lr=3e-3)
	
	for epoch in range(50):
		model_retrain, loss = retrain_sws_epoch(model_retrain, gmp, optimizer_kd, optimizer_gmp1, optimizer_gmp2, criterion, data_loader, tau)
		#if(writeTensorboard):
		#	for name, param in model_retrain.named_parameters():
		#	   writer.add_histogram(graph_title + "/" + name, param.clone().cpu().data.numpy(), epoch+1, bins='doane')

		if (trueAfterN(epoch, 10)):
			#gmp.print_batch = True
			print('Epoch: {}. Loss: {:.2f}'.format(epoch+1, float(loss.data)))
			layer_accuracy(model_retrain, gmp, model_orig, test_data_full, test_labels_full)
			
	exp_name = "{}_a{}_b{}_t{}m_{}".format(model_retrain.name, alpha, beta, tau, mixtures)
	torch.save(model_retrain, model_dir + '/mnist_layer_{}.m'.format(exp_name))
	with open(model_dir + '/mnist_retrain_{}_gmp.p'.format(exp_name),'wb') as f:
		pickle.dump(gmp, f)
			
	return model_retrain, gmp
	
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
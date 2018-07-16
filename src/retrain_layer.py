#execution example: python layer_retrain.py --layer 1 --alpha 2500 --beta 10 --tau 1e-6 --mixtures 4
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
import pickle
import argparse

model_dir = "./models/"
import model_archs
from utils_plot import show_sws_weights, show_weights, print_dims, prune_plot, draw_sws_graphs, joint_plot
from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty, layer_accuracy
from utils_misc import trueAfterN, logsumexp, root_dir, model_load_dir
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune, sws_prune_l2
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
from extract_targets import get_targets
retraining_epochs = 50

def retrain_layer(model_retrain, model_orig, data_loader, test_data_full, test_labels_full, alpha, beta, tau, mixtures, temp, data_size, savedir):
	
	weight_loader = model_retrain.state_dict()
	for layer in model_retrain.state_dict():
		weight_loader[layer] = model_orig.state_dict()[layer]
	model_retrain.load_state_dict(weight_loader)

	exp_name = "{}_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format(model_retrain.name, alpha, beta, retraining_epochs, tau, int(mixtures), int(temp), data_size)
	gmp = GaussianMixturePrior(mixtures, [x for x in model_retrain.parameters()], 0.99, ab = (alpha, beta), scaling = False)
	gmp.print_batch = False

	print ("Model Name: {}".format(model_retrain.name))
	criterion = nn.MSELoss()
	opt = torch.optim.Adam([
        {'params': model_retrain.parameters(), 'lr': 1e-4},
        {'params': [gmp.means], 'lr': 1e-4},
        {'params': [gmp.gammas, gmp.rhos], 'lr': 3e-3}])#log precisions and mixing proportions

	
	for epoch in range(retraining_epochs):
		model_retrain, loss = retrain_sws_epoch(model_retrain, gmp, opt, criterion, data_loader, tau, temp ** 2)

		if (trueAfterN(epoch, 10)):
			print('Epoch: {}. Loss: {:.2f}'.format(epoch+1, float(loss.data)))
			layer_accuracy(model_retrain, gmp, model_orig, test_data_full, test_labels_full)
			
	if(savedir!=""):
		torch.save(model_retrain, savedir + 'mnist_retrain_{}.m'.format(exp_name))
		with open(savedir + 'mnist_retrain_{}_gmp.p'.format(exp_name),'wb') as f:
			pickle.dump(gmp, f)
			
	return model_retrain, gmp

def get_layer_data(target_dir, temp, layer, data_size):
	x_start = 0
	x_end = 60000
	if (data_size == "search"):
		x_start = 40000
		x_end = 50000

	if (layer == 1):
		layer_model = model_archs.SWSModelConv1().cuda()
		input = Variable(train_data(fetch = "data")[x_start:x_end]).cuda()
		output = get_targets(target_dir, temp, ["conv1.out"])["conv1.out"][x_start:x_end]
	if (layer == 2):
		layer_model = model_archs.SWSModelConv2().cuda()
		input = nn.ReLU()(get_targets(target_dir, temp, ["conv1.out"])["conv1.out"][x_start:x_end])
		output = (get_targets(target_dir, temp, ["conv2.out"])["conv2.out"][x_start:x_end])
	if (layer == 3):
		layer_model = model_archs.SWSModelFC1().cuda()
		input = nn.ReLU()(get_targets(target_dir, temp, ["conv2.out"])["conv2.out"][x_start:x_end])
		output = get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end]
	if (layer == 4):
		layer_model = model_archs.SWSModelFC2().cuda()
		input = nn.ReLU()(get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end])
		output = get_targets(target_dir, temp, ["fc2.out"])["fc2.out"][x_start:x_end]

	dataset = torch.utils.data.TensorDataset(input.data, output.data)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	return layer_model, loader

def init_retrain_layer(alpha, beta, tau, mixtures, temp, data_size, layer, savedir = ""):
	test_data_full =  Variable(test_data(fetch = "data")).cuda()
	test_labels_full =  Variable(test_data(fetch = "labels")).cuda()
	val_data_full =  Variable(search_validation_data(fetch = "data")).cuda()
	val_labels_full =  Variable(search_validation_data(fetch = "labels")).cuda()

	model_name = "SWSModel"
	model_file = 'mnist_{}_{}_{}'.format(model_name, 100, data_size)
	model_orig = torch.load(model_load_dir + model_file + '.m').cuda()
	target_dir = model_file.replace("search", "full")

	layer_model, loader = get_layer_data(target_dir, temp, layer, data_size)

	model, gmp = retrain_layer(layer_model, model_orig, loader, test_data_full, test_labels_full, alpha, beta, tau, mixtures, temp, data_size, model_dir + model_file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--layer', dest = "layer", help = "Layer", required = True)
	parser.add_argument('--alpha', dest = "alpha", help = "Alpha", required = True)
	parser.add_argument('--beta', dest = "beta", help = "Beta", required = True)
	parser.add_argument('--tau', dest = "tau", help = "Tau", required = True)
	parser.add_argument('--mixtures', dest = "mixtures", help = "Number of mixtures", required = True)
	parser.add_argument('--temp', dest = "temp", help = "Temperature", required = False)
	parser.add_argument('--data', dest = "data", help = "Data to train on - 'full' training data (60k) or 'search' training data(50k)", required = True, choices = ('full','search'))
	parser.add_argument('--savedir', dest = "savedir", help = "Save Directory")
	args = parser.parse_args()
	alpha = float(args.alpha)
	beta = float(args.beta)
	tau = float(args.tau)
	layer = int(args.layer)
	mixtures = int(args.mixtures)
	if args.temp == None:
		temp = 0
	else:
		temp = float(args.temp)

	init_retrain_layer(alpha, beta, tau, mixtures, temp, args.data, layer, args.savedir)
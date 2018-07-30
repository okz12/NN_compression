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
from utils_plot import show_sws_weights, show_weights, print_dims, prune_plot, draw_sws_graphs, joint_plot, plot_data
from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty, layer_accuracy
from utils_misc import trueAfterN, logsumexp, root_dir, model_load_dir, get_ab
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune, sws_prune_l2, sws_prune_copy
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
from extract_targets import get_targets
retraining_epochs = 10

def retrain_layer(mean, var, zmean, zvar, mixtures, temp, tau, layer = 1, data_size = 'search', model_name = 'LeNet_300_100', loss_type = 'MSEHA', savedir = ""):
	ab = get_ab(mean, var)
	zab = get_ab(zmean, zvar)

	model_file = 'mnist_{}_{}_{}'.format(model_name, 100, data_size)
	full_model = torch.load(model_load_dir + model_file + ".m")
	layer_model, loader = get_layer_data(model_name, model_file, 0, layer, data_size)
	
	state_dict = copy.deepcopy(layer_model.state_dict())
	for layer in state_dict:
		state_dict[layer] = full_model.state_dict()[layer]
	layer_model.load_state_dict(state_dict)

	exp_name = "{}_m{}_zm{}_r{}_t{}_m{}_kdT{}_{}".format(layer_model.name, mean, zmean, retraining_epochs, tau, int(mixtures), int(temp), data_size)
	gmp = GaussianMixturePrior(mixtures, [x for x in layer_model.parameters()], 0.99, zero_ab = zab, ab = ab, scaling = False)
	gmp.print_batch = False
	criterion = nn.MSELoss()
	opt = torch.optim.Adam([
		{'params': layer_model.parameters(), 'lr': 1e-4},
		{'params': [gmp.means], 'lr': 3e-4},
		{'params': [gmp.gammas, gmp.rhos], 'lr': 3e-3}])#log precisions and mixing proportions

	res_stats = plot_data(layer_model, gmp, 'layer_retrain', full_model, data_size, "CE", (mean, var), (zmean, zvar), tau, temp, mixtures)
	for epoch in range(retraining_epochs):

		layer_model, loss = retrain_sws_epoch(layer_model, gmp, opt, loader, tau, temp, loss_type)
		res_stats.data_epoch(epoch+1, layer_model, gmp)

		if (trueAfterN(epoch, 25)):
			print('Epoch: {}. Loss: {:.2f}'.format(epoch+1, float(loss.data)))
			#show_sws_weights(model = layer_model, means = list(gmp.means.data.clone().cpu()), precisions = list(gmp.gammas.data.clone().cpu()))
			#res = layer_accuracy(layer_model, gmp, full_model, test_data_full, test_labels_full)
	prune_model = sws_prune_copy(layer_model, gmp)
	res_stats.data_prune(layer_model)
	
	res_test = layer_accuracy(layer_model, gmp, res_stats.full_model, res_stats.test_data_full, res_stats.test_labels_full)
	res = res_stats.gen_dict()
	res['compress_test'] = res_test[0]
	res['prune_test'] = res_test[2]
	res['sparsity'] = res_test[3]
	if (data_size == "search"):
		res_val = layer_accuracy(layer_model, gmp, res_stats.full_model, res_stats.val_data_full, res_stats.val_labels_full)
		res['prune_val'] = res_val[2]
		res['compress_val'] = res_val[0]
	
	
	if (savedir != ""):
		torch.save(model, savedir + '/mnist_retrain_layer_model_{}.m'.format(exp_name))
		with open(savedir + '/mnist_retrain_layer_gmp_{}.p'.format(exp_name),'wb') as f:
			pickle.dump(gmp, f)
		with open(savedir + '/mnist_retrain_layer_res_{}.p'.format(exp_name),'wb') as f:
			pickle.dump(res, f)
	return layer_model, gmp, res

def get_layer_data(model_name, target_dir, temp, layer, data_size):
	x_start = 0
	x_end = 60000
	if (data_size == "search"):
		x_start = 40000
		x_end = 50000
	if (model_name == "SWSModel"):
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

	if (model_name == "LeNet_300_100"):
		if (layer == 1):
			layer_model = model_archs.LeNet_300_100FC1().cuda()
			input = Variable(train_data(fetch = "data")[x_start:x_end]).cuda()
			output = get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end]
		if (layer == 2):
			layer_model = model_archs.LeNet_300_100FC2().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end])
			output = (get_targets(target_dir, temp, ["fc2.out"])["fc2.out"][x_start:x_end])
		if (layer == 3):
			layer_model = model_archs.LeNet_300_100FC3().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["fc2.out"])["fc2.out"][x_start:x_end])
			output = get_targets(target_dir, temp, ["fc3.out"])["fc3.out"][x_start:x_end]


	dataset = torch.utils.data.TensorDataset(input.data, output.data)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	return layer_model, loader

def init_retrain_layer(alpha, beta, tau, temp, mixtures, model_name, data_size, layer, savedir = "", loss_type = 'MSEHNA'):
	test_data_full =  Variable(test_data(fetch = "data")).cuda()
	test_labels_full =  Variable(test_data(fetch = "labels")).cuda()
	val_data_full =  Variable(search_validation_data(fetch = "data")).cuda()
	val_labels_full =  Variable(search_validation_data(fetch = "labels")).cuda()

	model_file = 'mnist_{}_{}_{}'.format(model_name, 100, data_size)
	model_orig = torch.load(model_load_dir + model_file + '.m').cuda()
	target_dir = model_file.replace("search", "full")

	layer_model, loader = get_layer_data(target_dir, temp, layer, data_size)

	model, gmp = retrain_layer(layer_model, model_orig, loader, test_data_full, test_labels_full, alpha, beta, tau, mixtures, temp, loss_type, data_size, model_dir + model_file)
	return model, gmp

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
	parser.add_argument('--model', dest = "model", help = "Model to train", required = True, choices = ('SWSModel', 'LeNet_300_100'))
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

	init_retrain_layer(alpha, beta, tau, temp, mixtures, args.model, args.data, layer, args.savedir)
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

model_dir = "./models/"
import models
from utils_plot import show_sws_weights, show_weights, print_dims, prune_plot, draw_sws_graphs, joint_plot
from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty, retrain_layer, layer_accuracy
from utils_misc import trueAfterN, logsumexp
from utils_Sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--layer', dest = "layer", help = "Layer", required = True)
parser.add_argument('--alpha', dest = "alpha", help = "Alpha", required = True)
parser.add_argument('--beta', dest = "beta", help = "Beta", required = True)
parser.add_argument('--tau', dest = "tau", help = "Tau", required = True)
parser.add_argument('--mixtures', dest = "mixtures", help = "Number of mixtures", required = True)
parser.add_argument('--temp', dest = "temp", help = "Temperature", required = False)
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

batch_size = 256
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False , transform=transforms.ToTensor(), download=True)
test_data_full = Variable(test_dataset.test_data.float()/255.0).cuda()
test_labels_full = Variable(test_dataset.test_labels).cuda()

model_name = "SWSModel"
training_epochs = 100
model_file = 'mnist_{}_{}'.format(model_name, training_epochs)
model_orig = torch.load("{}{}.m".format(model_dir, model_file)).cuda()

if (layer == 1):
	layer_model = models.SWSModelConv1().cuda()
	input = (train_dataset.train_data.float()/255.0).cuda()
	output = torch.load("{}{}/{}.out.m".format(model_dir, model_file, "conv1")).data
if (layer == 2):
	layer_model = models.SWSModelConv2().cuda()
	input = nn.ReLU()(torch.load("{}{}/{}.out.m".format(model_dir, model_file, "conv1")).data)
	output = torch.load("{}{}/{}.out.m".format(model_dir, model_file, "conv2")).data
if (layer == 3):
	layer_model = models.SWSModelFC1().cuda()
	input = nn.ReLU()(torch.load("{}{}/{}.out.m".format(model_dir, model_file, "conv2")).data)
	output = torch.load("{}{}/{}.out.m".format(model_dir, model_file, "fc1")).data
	if temp != 0:
		output = nn.Softmax(dim=1)(output/T)
if (layer == 4):
	layer_model = models.SWSModelFC2().cuda()
	input = torch.load("{}{}/{}.out.m".format(model_dir, model_file, "fc1")).data
	output = torch.load("{}{}/{}.out.m".format(model_dir, model_file, "fc2")).data
	if temp != 0:
		input = nn.Softmax(dim=1)(input/T)
		output = nn.Softmax(dim=1)(output/T)
	input = nn.ReLU()(input)

dataset = torch.utils.data.TensorDataset(input, output)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

_ = retrain_layer(layer_model, model_orig, loader, test_data_full, test_labels_full, alpha, beta, tau, mixtures, model_dir + model_file)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

model_dir = "./models/"
import models
from plot_utils import show_sws_weights, show_weights, print_dims, prune_plot, draw_sws_graphs, joint_plot
from model_utils import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty
from helpers import trueAfterN, logsumexp
from sws_utils import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune
import copy
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', dest = "alpha", help="Gamma Prior Alpha", required=True)
parser.add_argument('--beta', dest = "beta", help="Gamma Prior Beta", required=True)
parser.add_argument('--tau', dest = "tau", help="Tau: Complexity and Error Loss trade-off parameter", required=True)
parser.add_argument('--model', dest = "model", help="Tau: Complexity and Error Loss trade-off parameter", required=True)
parser.add_argument('--mixtures', dest = "mixtures", help="Mixtures: Number of Gaussian prior mixtures", required=True)
args = parser.parse_args()
alpha = float(args.alpha)
beta = float(args.beta)
tau = float(args.tau)
model_name = args.model

#Data
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False , transform=transforms.ToTensor(), download=True)

batch_size = 256
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False , transform=transforms.ToTensor(), download=True)
test_data_full = Variable(test_dataset.test_data.float()/255.0).cuda()
test_labels_full = Variable(test_dataset.test_labels).cuda()

model = torch.load(model_dir + 'mnist_{}_{}.m'.format(model_name, 100)).cuda()
gmp = GaussianMixturePrior(16, [x for x in model.parameters()], 0.99, ab = (alpha, beta))

sws_param1 = [gmp.means]
sws_param2 = [gmp.gammas, gmp.rhos]

optimizer_gmp = torch.optim.SGD(sws_param1, lr=1e-4)
optimizer_gmp2 = torch.optim.SGD(sws_param2, lr=3e-3)

retraining_epochs = 50

exp_name = "a{}_b{}_r{}_t{}".format(alpha, beta, retraining_epochs, tau)

if args.temp == None:
	temp = 0
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
else:
	temp = float(args.temp)
	criterion = nn.MSELoss()
	optimizer_kd_1 = torch.optim.Adam(model_retrain.conv1.parameters(), lr=1e-4)
	optimizer_kd_2 = torch.optim.Adam(model_retrain.conv2.parameters(), lr=1e-4)
	optimizer_kd_3 = torch.optim.Adam(model_retrain.fc1.parameters(), lr=1e-4)
	optimizer_kd_4 = torch.optim.Adam(model_retrain.fc2.parameters(), lr=1e-4)
	output = torch.load("{}{}/{}.out.m".format(model_dir, model_file, "fc2")).data
	output = nn.Softmax(dim=1)(output/T)
	dataset = torch.utils.data.TensorDataset(input, output)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

if model_name == "SWSModel":
	model = models.SWSModel().cuda()
else:
	model = models.LeNet_300_100().cuda()

for epoch in range(retraining_epochs):
    print("Epoch: {}".format(epoch+1))
    model, loss = retrain_sws_epoch(model, gmp, optimizer, optimizer_gmp, optimizer_gmp2, criterion, loader, tau)
	
	model, loss = retrain_sws_epoch(model, gmp, optimizer_kd_1, optimizer_gmp1, optimizer_gmp2, criterion, loader, tau)
    model, loss = retrain_sws_epoch(model, gmp, optimizer_kd_2, optimizer_gmp1, optimizer_gmp2, criterion, loader, tau)
    model, loss = retrain_sws_epoch(model, gmp, optimizer_kd_3, optimizer_gmp1, optimizer_gmp2, criterion, loader, tau)
    model, loss = retrain_sws_epoch(model, gmp, optimizer_kd_4, optimizer_gmp1, optimizer_gmp2, criterion, loader, tau)

    test_acc = test_accuracy(test_data_full, test_labels_full, model)
    train_acc = test_accuracy(train_data_full, train_labels_full, model)
	
    if (trueAfterN(epoch, 10)):
        test_acc = test_accuracy(test_data_full, test_labels_full, model)
		print('Epoch: {}. Test Accuracy: {:.2f}'.format(epoch+1, test_acc[0]))

torch.save(model, model_dir + 'mnist_retrain_m{}_a{}_b{}_r{}.m'.format(model.name, alpha, beta, retraining_epochs))
with open(model_dir + 'mnist_retrain_m{}_a{}_b{}_r{}_gmp.p'.format(model.name, alpha, beta, retraining_epochs),'wb') as f:
    pickle.dump(gmp, f)
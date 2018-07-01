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

def show_weights(model):
	"""
	shows histograms of 3 weight layers in a model - built for LeNet 300-100
	"""
	weight_list = [x for x in model.state_dict().keys() if 'weight' in x]
	plt.clf()
	plt.figure(figsize=(18, 3))
	for i,weight in enumerate(weight_list):
		plt.subplot(131 + i)
		fc_w = model.state_dict()[weight]
		sns.distplot(fc_w.view(-1).cpu().numpy())
		plt.title('Layer: {}'.format(weight))
	plt.show()
	
def print_dims(model):
	"""
	print dimensions of a model
	"""
	for i,params in enumerate(model.parameters()):
		param_list = []
		for pdim in params.size():
			param_list.append(str(pdim))
		if i%2==0:
			dim_str = "x".join(param_list)
		else:
			print (dim_str + " + " + "x".join(param_list))
			
def prune_plot(temp, dev_res, perc_res, test_acc_o, train_acc_o, weight_penalty_o, test_acc_kd, train_acc_kd, weight_penalty_kd):
	"""
	KD pruning plots for standard deviation and percentile based weight pruning
	-- deprecated after SWS pruning has been implemented
	"""
	c1 = '#2ca02c'
	c2 = '#1f77b4'
	c3 = '#ff7f0e'
	c4 = '#d62728'
	plt.clf()
	ncols = 5
	nrows = 1

	plt.figure(figsize=(25,4))
	plt.subplot(nrows, ncols, 1)
	plt.plot(perc_res['pruned'], perc_res['train ce'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['train ce'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=train_acc_o[1], label="Original", color = c3, linestyle='--')
	plt.axhline(y=train_acc_kd[1], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Cross Entropy Loss")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=2)
	plt.title("Train CE Loss")

	plt.subplot(nrows, ncols, 2)
	plt.plot(perc_res['pruned'], perc_res['test ce'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['test ce'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=test_acc_o[1], label="Original", color = c3, linestyle='--')
	plt.axhline(y=test_acc_kd[1], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Cross Entropy Loss")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=2)
	plt.title("Test CE Loss")

	plt.subplot(nrows, ncols, 3)
	plt.plot(perc_res['pruned'], perc_res['train acc'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['train acc'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=train_acc_o[0], label="Original", color = c3, linestyle='--')
	plt.axhline(y=train_acc_kd[0], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Accuracy(%)")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=6)
	plt.title("Train Accuracy")

	plt.subplot(nrows, ncols, 4)
	plt.plot(perc_res['pruned'], perc_res['test acc'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['test acc'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=test_acc_o[0], label="Original", color = c3, linestyle='--')
	plt.axhline(y=test_acc_kd[0], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Accuracy(%)")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=6)
	plt.title("Test Accuracy")

	plt.subplot(nrows, ncols, 5)
	plt.plot(perc_res['pruned'], perc_res['L2'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['L2'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=weight_penalty_o, label="Original", color = c3, linestyle='--')
	plt.axhline(y=weight_penalty_kd, label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("L2")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=6)
	plt.title("Model L2")
	plt.show()
	
	
###
def show_sws_weights(model, means=0, precisions=0, epoch=-1, accuracy=-1, savefile = ""):
	"""
	show model weight histogram with mean and precisions
	"""
	weights = np.array([], dtype=np.float32)
	for layer in model.state_dict():
		weights = np.hstack( (weights, model.state_dict()[layer].view(-1).cpu().numpy()) )
		
	plt.clf()
	plt.figure(figsize=(20, 6))
	
	#1 - Non-log plot
	plt.subplot(2,1,1)
	
	#Title
	if (epoch !=-1 and accuracy == -1):
		plt.title("Epoch: {:0=3d}".format(epoch+1))
	if (accuracy != -1 and epoch == -1):
		plt.title("Accuracy: {:.2f}".format(accuracy))
	if (accuracy != -1 and epoch != -1):
		plt.title("Epoch: {:0=3d} - Accuracy: {:.2f}".format(epoch+1, accuracy))
	
	sns.distplot(weights, kde=False, color="g",bins=200,norm_hist=True, hist_kws={'log':False})
	
	#plot mean and precision
	if not (means==0 or precisions==0):
		plt.axvline(0, linewidth = 1)
		std_dev0 = np.sqrt(1/np.exp(precisions[0]))
		plt.axvspan(xmin=-std_dev0, xmax=std_dev0, alpha=0.3)

		for mean, precision in zip(means, precisions[1:]):
			plt.axvline(mean, linewidth = 1)
			std_dev = np.sqrt(1/np.exp(precision))
			plt.axvspan(xmin=mean - std_dev, xmax=mean + std_dev, alpha=0.1)
	
	#plt.xticks([])
	#plt.xlabel("Weight Value")
	plt.ylabel("Density")
	
	plt.xlim([-1, 1])
	plt.ylim([0, 60])
	
	#2-Logplot
	plt.subplot(2,1,2)
	sns.distplot(weights, kde=False, color="g",bins=200,norm_hist=True, hist_kws={'log':True})
	#plot mean and precision
	if not (means==0 or precisions==0):
		plt.axvline(0, linewidth = 1)
		std_dev0 = np.sqrt(1/np.exp(precisions[0]))
		plt.axvspan(xmin=-std_dev0, xmax=std_dev0, alpha=0.3)

		for mean, precision in zip(means, precisions[1:]):
			plt.axvline(mean, linewidth = 1)
			std_dev = np.sqrt(1/np.exp(precision))
			plt.axvspan(xmin=mean - std_dev, xmax=mean + std_dev, alpha=0.1)
	plt.xlabel("Weight Value")
	plt.ylabel("Density")
	plt.xlim([-1, 1])
	plt.ylim([1e-3, 1e2])
	
	if savefile!="":
		plt.savefig("./figs/{}_{}.png".format(savefile, epoch+1), bbox_inches='tight')
		plt.close()
	else:
		plt.show()
		
		
###
def draw_sws_graphs(means = -1, stddev = -1, mixprop = -1, acc = -1, savefile=""):
	"""
	plot showing evolution of sws retraining
	"""
	plt.clf()
	plt.figure(figsize=(20, 10))
	plt.subplot(2,2,1)
	plt.plot(means)
	plt.title("Mean")
	plt.xlim([0, means.shape[0]-1])
	plt.xlabel("Epoch")

	plt.subplot(2,2,2)
	plt.plot(mixprop[:,1:])
	plt.yscale("log")
	plt.title("Mixing Proportions")
	plt.xlim([0, mixprop.shape[0]-1])
	plt.xlabel("Epoch")

	plt.subplot(2,2,3)
	plt.plot(stddev[:,1:])
	plt.yscale("log")
	plt.title("Standard Deviations")
	plt.xlim([0, stddev.shape[0]-1])
	plt.xlabel("Epoch")

	plt.subplot(2,2,4)
	plt.plot(acc)
	plt.title("Accuracy")
	plt.xlim([0, acc.shape[0]-1])
	plt.xlabel("Epoch")
	plt.show()
	
	if savefile!="":
		plt.savefig("./exp/{}.png".format(savefile), bbox_inches='tight')
		plt.close()
	else:
		plt.show()
		
		
def joint_plot(model, model_orig, gmp, epoch, retraining_epochs, acc, savefile = ""):
	"""
	joint distribution plot weights before and after sws retraining
	"""
	weights_T = np.array([], dtype=np.float32)
	for layer in model.state_dict():
		weights_T = np.hstack( (weights_T, model.state_dict()[layer].view(-1).cpu().numpy()) )

	weights_0 = np.array([], dtype=np.float32)
	for layer in model_orig.state_dict():
		weights_0 = np.hstack( (weights_0, model_orig.state_dict()[layer].view(-1).cpu().numpy()) )

	#get mean, stddev
	mu_T = np.concatenate([np.zeros(1), gmp.means.clone().data.cpu().numpy()])
	std_T = np.sqrt(1/np.exp(gmp.gammas.clone().data.cpu().numpy()))

	x0 = -1.2
	x1 = 1.2
	I = np.random.permutation(len(weights_0))
	f = sns.jointplot(weights_0[I], weights_T[I], size=8, kind="scatter", color="b", stat_func=None, edgecolor='w',
					  marker='o', joint_kws={"s": 8}, marginal_kws=dict(bins=1000), ratio=4)
	f.ax_joint.hlines(mu_T, x0, x1, lw=0.5)

	for k in range(len(mu_T)):
		if k == 0:
			f.ax_joint.fill_between(np.linspace(x0, x1, 10), mu_T[k] - 2 * std_T[k], mu_T[k] + 2 * std_T[k],
									color='g', alpha=0.1)
		else:
			f.ax_joint.fill_between(np.linspace(x0, x1, 10), mu_T[k] - 2 * std_T[k], mu_T[k] + 2 * std_T[k],
									color='b', alpha=0.1)
	
	plt.title("Epoch: %d /%d\nTest accuracy: %.4f " % (epoch+1, retraining_epochs, acc))
	f.ax_marg_y.set_xscale("log")
	f.set_axis_labels("Pretrained", "Retrained")
	f.ax_marg_x.set_xlim(-1, 1)
	f.ax_marg_y.set_ylim(-1, 1)
	if savefile!="":
		plt.savefig("./figs/jp_{}_{}.png".format(savefile, epoch+1), bbox_inches='tight')
		plt.close()
	else:
		plt.show()
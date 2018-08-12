#execution example: python retrain.py --model SWSModel --alpha 2500 --beta 10 --tau 1e-6 --mixtures 8 --temp 10
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import numpy as np

import model_archs
from utils_plot import show_sws_weights, show_weights, print_dims, prune_plot, draw_sws_graphs, joint_plot, plot_data
from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty
from utils_misc import trueAfterN, logsumexp, root_dir, model_load_dir, get_ab, get_sparsity
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune, sws_prune_l2, sws_prune_copy, compressed_model
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
import copy
import pickle
import argparse
retraining_epochs = 50

def retrain_model(mean, var, zmean, zvar, tau, temp, mixtures, model_name, data_size, loss_type = 'MSESNT', scaling = False, model_save_dir = "",  fn="", dset="mnist"):
	ab = get_ab(mean, var)
	zab = get_ab(zmean, zvar)

	if(data_size == 'search'):
		train_dataset = search_retrain_data
		val_data_full = Variable(search_validation_data(fetch='data', dset=dset)).cuda()
		val_labels_full = Variable(search_validation_data(fetch='labels', dset=dset)).cuda()
		(x_start, x_end) = (40000, 50000)
	if(data_size == 'full'):
		train_dataset = train_data
		(x_start, x_end) = (0, 60000)
	test_data_full = Variable(test_data(fetch='data', dset=dset)).cuda()
	test_labels_full = Variable(test_data(fetch='labels', dset=dset)).cuda()
		
	model_file = '{}_{}_{}_{}'.format(dset, model_name, 100, data_size)
	model = torch.load(model_load_dir + model_file + '.m').cuda()
		
	if temp == 0:
		loader = torch.utils.data.DataLoader(dataset=train_dataset(onehot = (loss_type == 'MSESNT'), dset=dset), batch_size=batch_size, shuffle=True)
	else:
		output = torch.load("{}{}_targets/{}.out.m".format(model_load_dir, model_file, "fc3" if "300_100" in model.name else "fc2"))[x_start:x_end]###
		output = (nn.Softmax(dim=1)(output/temp)).data
		dataset = torch.utils.data.TensorDataset(train_dataset(fetch='data', dset=dset), output)
		loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	criterion = nn.CrossEntropyLoss()###
	s = "s" if scaling else "f" 
	exp_name = "{}_m{}_zm{}_r{}_t{}_m{}_kdT{}_{}_{}".format(model.name, mean, zmean, retraining_epochs, tau, int(mixtures), int(temp), s, data_size) + fn
	gmp = GaussianMixturePrior(mixtures, [x for x in model.parameters()], 0.99, zero_ab = zab, ab = ab, scaling = scaling)
	gmp.print_batch = False

	mlr = 0.5e-4 if scaling else 0.5e-4

	optimizable_params = [
		{'params': model.parameters(), 'lr': 2e-4},
		{'params': [gmp.means], 'lr': mlr},
		{'params': [gmp.gammas, gmp.rhos], 'lr': 3e-3}]
	if (scaling):
		if "SWS" in model_name:
			slr = 2e-5
		else:
			slr = 1e-6
		optimizable_params = optimizable_params + [{'params': gmp.scale, 'lr': slr}]
	
	opt = torch.optim.Adam(optimizable_params)#log precisions and mixing proportions

	res_stats = plot_data(init_model = model, gmp = gmp, mode = 'retrain', data_size = data_size, loss_type='CE', mv = (mean, var), zmv = (zmean, zvar), tau = tau, temp = temp, mixtures = mixtures, dset = dset)
	s_hist = []
	a_hist = []
	for epoch in range(retraining_epochs):
		### [ACT DISABLE LR]
		#if(scaling and epoch == 0):
		#	opt.param_groups[3]['lr'] = 0
		#	print ("Scaling Disabled - Epoch {}".format(epoch))
		model, loss = retrain_sws_epoch(model, gmp, opt, loader, tau, temp, loss_type)
		res_stats.data_epoch(epoch + 1, model, gmp)
		

		if (trueAfterN(epoch, 10)):
			#test_acc = test_accuracy(test_data_full, test_labels_full, model)
			nm = sws_prune_copy(model, gmp)
			s = get_sparsity(nm)
			a = test_accuracy(test_data_full, test_labels_full, nm)[0]

			print('Epoch: {}. Test Accuracy: {:.2f}, Prune Accuracy: {:.2f}, Sparsity: {:.2f}'.format(epoch+1, res_stats.test_accuracy[-1], a, s))
			#show_sws_weights(model = model, means = list(gmp.means.data.clone().cpu()), precisions = list(gmp.gammas.data.clone().cpu()), epoch = epoch)###
		nm = sws_prune_copy(model, gmp)
		s = get_sparsity(nm)
		a = test_accuracy(test_data_full, test_labels_full, nm)[0]
		s_hist.append(s)
		a_hist.append(a)

		if (data_size == 'search' and (epoch>12) and trueAfterN(epoch, 2)):
			val_acc =  res_stats.test_accuracy[-1]
			if (val_acc < 50.0):
				print ("Terminating Search - Epoch: {} - Val Acc: {:.2f}".format(epoch, val_acc))
				break
	
	res = res_stats.gen_dict()
	
	model_prune = sws_prune_copy(model, gmp)

	res_stats.data_prune(model_prune)
	res = res_stats.gen_dict()
	res['test_prune_acc'] = a_hist
	res['test_prune_sp'] = s_hist
	cm = compressed_model(model_prune.state_dict(), [gmp])
	res['cm'] = cm.get_cr_list()

	if (data_size == "search"):
		print('Retrain Test: {:.2f}, Retrain Validation: {:.2f}, Prune Test: {:.2f}, Prune Validation: {:.2f}, Prune Sparsity: {:.2f}'
		.format(res['test_acc'][-1], res['val_acc'][-1], res['prune_acc']['test'], res['prune_acc']['val'], res['sparsity']))
	else:
		print('Retrain Test: {:.2f}, Prune Test: {:.2f}, Prune Sparsity: {:.2f}'.format(res['test_acc'][-1], res['prune_acc']['test'],res['sparsity']))

	if(model_save_dir!=""):
		torch.save(model, model_save_dir + '/{}_retrain_model_{}.m'.format(dset, exp_name))
		with open(model_save_dir + '/{}_retrain_gmp_{}.p'.format(dset, exp_name),'wb') as f:
			pickle.dump(gmp, f)
		with open(model_save_dir + '/{}_retrain_res_{}.p'.format(dset, exp_name),'wb') as f:
			pickle.dump(res, f)

	return model, gmp, res
   

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--alpha', dest = "alpha", help="Gamma Prior Alpha", required=True, type=(float))
	parser.add_argument('--beta', dest = "beta", help="Gamma Prior Beta", required=True, type=(float))
	parser.add_argument('--tau', dest = "tau", help="Tau: Complexity and Error Loss trade-off parameter", required=True, type=(float))
	parser.add_argument('--temp', dest = "temp", help="Temperature: Final softmax temperature for knowledge distillation", required=False, type=(int))
	parser.add_argument('--mixtures', dest = "mixtures", help="Mixtures: Number of Gaussian prior mixtures", required=True, type=(int))
	parser.add_argument('--model', dest = "model", help = "Model to train", required = True, choices = ('SWSModel', 'LeNet_300_100'))
	parser.add_argument('--data', dest = "data", help = "Data to train on - 'full' training data (60k) or 'search' training data(50k)", required = True, choices = ('full','search'))
	parser.add_argument('--savedir', dest = "savedir", help = "Save Directory")
	parser.add_argument('--scale', dest = "scale", help = "Allow different scales on each layer", action = 'store_true')
	args = parser.parse_args()
	alpha = float(args.alpha)
	beta = float(args.beta)
	tau = float(args.tau)
	mixtures = int(args.mixtures)
	model_name = args.model
	if args.temp == None:
		temp = 0
	else:
		temp = float(args.temp)
	scaling = True if args.scale != None else False
		
	retrain_model(alpha, beta, tau, temp, mixtures, model_name, args.data, args.savedir, args.scale)
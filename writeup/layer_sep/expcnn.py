import sys
sys.path.insert(0,'../../src/')

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
from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty, layer_accuracy
from utils_plot import plot_data
from utils_misc import trueAfterN, logsumexp, root_dir, model_load_dir, get_ab, get_sparsity
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune, sws_prune_l2, sws_prune_copy, sws_replace, compressed_model
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
from extract_targets import get_targets, get_layer_data
from retrain_layer import retrain_layer
retraining_epochs = 50


test_data_full = Variable(test_data(fetch = "data")).cuda()
test_labels_full = Variable(test_data(fetch = "labels")).cuda()
#val_data_full = Variable(search_validation_data(fetch = "data")).cuda()
#val_labels_full = Variable(search_validation_data(fetch = "labels")).cuda()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dset', dest = "dset", help = "Path to model to extract from", required = True)
args = parser.parse_args()
dset = args.dset



scaling = False
res_str = ""
res_list = []
model_save_dir = "./files"
tau_list = [4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 10e-5]
if (dset == "fashionmnist"):
    tau_list = []
for tau in tau_list:
    model_name = "SWSModel"
    data_size = "full"
    model_file = '{}_{}_{}_{}'.format(dset, model_name, 100, data_size)
    model = torch.load(model_load_dir + model_file + '.m').cuda()

    targets_dict = get_targets(model_file)
    inputs = train_data(fetch = "data", dset = dset).cuda()
    targets = torch.cat((targets_dict['conv1.out'].view(60000, -1),targets_dict['conv2.out'].view(60000, -1),targets_dict['fc1.out'].view(60000, -1),targets_dict['fc2.out'].view(60000, -1)), 1).data.cuda()
    if data_size == "search":
        inputs = inputs[0:10000]
        targets = targets[0:10000]

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    #Get targets at each layer
    gmp = GaussianMixturePrior(16, [x for x in model.parameters()], 0.99, zero_ab = get_ab(2000, 1000), ab = get_ab(100,10), scaling = scaling)
    opt_1 = torch.optim.Adam([{'params': model.conv1.parameters(), 'lr': 2e-4}])
    opt_2 = torch.optim.Adam([{'params': model.conv2.parameters(), 'lr': 2e-4}])
    opt_3 = torch.optim.Adam([{'params': model.fc1.parameters(), 'lr': 2e-4}])
    opt_4 = torch.optim.Adam([{'params': model.fc2.parameters(), 'lr': 2e-4}])
    if (not scaling):
        opt_gmp = torch.optim.Adam([{'params': [gmp.means], 'lr': 0.5e-4}, {'params': [gmp.gammas, gmp.rhos], 'lr': 3e-3}])
    else:
        opt_gmp = torch.optim.Adam([{'params': [gmp.means], 'lr': 0.5e-4}, {'params': [gmp.gammas, gmp.rhos], 'lr': 3e-3}, {'params': [gmp.scale], 'lr': 1e-6}])
        
        
    mean = 100
    var = 10
    zmean = 2000
    zvar = 1000
    temp = 1
    mixtures = 16
    exp_name = "{}_m{}_zm{}_r{}_t{}_m{}_kdT{}_{}_{}".format(model.name, mean, zmean, retraining_epochs, tau, int(mixtures), int(temp), "f", "full")
    res_stats = plot_data(init_model = model, gmp = gmp, mode = 'retrain', data_size = data_size, loss_type='CE', mv = (100, 10), zmv = (zmean, zvar), tau = tau, temp = temp, mixtures = mixtures, dset = dset)
    s_hist = []
    a_hist = []


    for epoch in range(retraining_epochs):
        for i, (images, targets) in enumerate(loader):
            images=images.cuda()
            targets=targets.cuda()
            images = Variable(images)
            targets = Variable(targets)    
            opt_1.zero_grad()
            opt_2.zero_grad()
            opt_3.zero_grad()
            opt_4.zero_grad()
            opt_gmp.zero_grad()

            forward = model.layer_forward(images)
            loss_acc = nn.MSELoss()(forward, targets)

            loss = loss_acc + tau * gmp.call()
            loss.backward()

            opt_1.step()
            opt_2.step()
            opt_3.step()
            opt_3.step()
            opt_gmp.step()
            
        res_stats.data_epoch(epoch + 1, model, gmp)
        nm = sws_prune_copy(model, gmp)
        s = get_sparsity(nm)
        a = test_accuracy(test_data_full, test_labels_full, nm)[0]
        s_hist.append(s)
        a_hist.append(a)
        if (trueAfterN(epoch, 25)):
            test_acc = test_accuracy(test_data_full, test_labels_full, model)
            prune_model = sws_prune_copy(model, gmp)
            prune_acc = test_accuracy(test_data_full, test_labels_full, prune_model)
            sparsity = get_sparsity(prune_model)
            res_list.append((tau, test_acc[0], prune_acc[0], sparsity))
            res_str += 'Tau: {}, Epoch: {}. Test Accuracy: {:.2f} Prune Accuracy: {:.2f} Sparsity: {:.2f}\n'.format(tau, epoch+1, test_acc[0], prune_acc[0], sparsity)
            print('Tau: {}, Epoch: {}. Test Accuracy: {:.2f} Prune Accuracy: {:.2f} Sparsity: {:.2f}'.format(tau, epoch+1, test_acc[0], prune_acc[0], sparsity))
            print(res_list)
            #show_sws_weights(model = model, means = list(gmp.means.data.clone().cpu()), precisions = list(gmp.gammas.data.clone().cpu()), epoch = epoch)
    model_prune = sws_prune_copy(model, gmp)

    res_stats.data_prune(model_prune)
    res = res_stats.gen_dict()
    res['test_prune_acc'] = a_hist
    res['test_prune_sp'] = s_hist
    if(model_save_dir!=""):
        torch.save(model, model_save_dir + '/{}_retrain_model_{}.m'.format(dset, exp_name))
        with open(model_save_dir + '/{}_retrain_gmp_{}.p'.format(dset, exp_name),'wb') as f:
            pickle.dump(gmp, f)
        with open(model_save_dir + '/{}_retrain_res_{}.p'.format(dset, exp_name),'wb') as f:
            pickle.dump(res, f)
print (res_str)
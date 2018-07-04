#execution example: python retrain.py --model SWSModel --alpha 2500 --beta 10 --tau 1e-6 --mixtures 8 --temp 10
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import numpy as np

import model_archs
from utils_plot import show_sws_weights, show_weights, print_dims, prune_plot, draw_sws_graphs, joint_plot
from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty
from utils_misc import trueAfterN, logsumexp, root_dir, model_load_dir
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
import copy
import pickle
import argparse
retraining_epochs = 50

def retrain_model(alpha, beta, tau, temp, mixtures, model_name, data_size, model_save_dir = ""):
    if(data_size == 'search'):
        train_dataset = search_retrain_data
        val_data_full = Variable(search_validation_data(fetch='data')).cuda()
        val_labels_full = Variable(search_validation_data(fetch='labels')).cuda()
        (x_start, x_end) = (40000, 50000)
    if(data_size == 'full'):
        train_dataset = train_data
        (x_start, x_end) = (0, 60000)
    test_data_full = Variable(test_data(fetch='data')).cuda()
    test_labels_full = Variable(test_data(fetch='labels')).cuda()
        
    model_file = 'mnist_{}_{}_{}'.format(model_name, 100, data_size)
    model = torch.load(model_load_dir + model_file + '.m').cuda()
        
    if temp == 0:
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(dataset=train_dataset(), batch_size=batch_size, shuffle=True)
    else:
        criterion = nn.MSELoss()
        output = torch.load("{}{}_targets/{}.out.m".format(model_load_dir, model_file.replace("search", "full"), "fc2"))[x_start:x_end]
        output = (nn.Softmax(dim=1)(output/temp)).data
        dataset = torch.utils.data.TensorDataset(train_dataset(fetch='data'), output)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    exp_name = "m{}_a{}_b{}_r{}_t{}_kdT{}_{}".format(model.name, alpha, beta, retraining_epochs, tau, temp, data_size)
    gmp = GaussianMixturePrior(mixtures, [x for x in model.parameters()], 0.99, ab = (alpha, beta))
    gmp.print_batch = False

    sws_param1 = [gmp.means]
    sws_param2 = [gmp.gammas, gmp.rhos]

    opt = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': sws_param1, 'lr': 1e-4},
        {'params': sws_param2, 'lr': 3e-3}])

    for epoch in range(retraining_epochs):
        model, loss = retrain_sws_epoch(model, gmp, opt, criterion, loader, tau)

        if (trueAfterN(epoch, 10)):
            test_acc = test_accuracy(test_data_full, test_labels_full, model)
            print('Epoch: {}. Test Accuracy: {:.2f}'.format(epoch+1, test_acc[0]))
    if(model_save_dir!=""):
        torch.save(model, model_save_dir + '/mnist_retrain_{}.m'.format(exp_name))
        with open(model_save_dir + '/mnist_retrain_{}_gmp.p'.format(exp_name),'wb') as f:
            pickle.dump(gmp, f)
    
    test_accuracy_pre = float((test_accuracy(test_data_full, test_labels_full, model)[0]))
    val_accuracy_pre = 0 if (data_size != 'search') else float((test_accuracy(val_data_full, val_labels_full, model)[0]))
    
    model_prune = copy.deepcopy(model)
    model_prune.load_state_dict(sws_prune(model_prune, gmp))
    prune_acc = (test_accuracy(test_data_full, test_labels_full, model_prune))
    test_accuracy_prune = float((test_accuracy(test_data_full, test_labels_full, model_prune)[0]))
    val_accuracy = 0 if (data_size != 'search') else float((test_accuracy(val_data_full, val_labels_full, model_prune)[0]))
    sparsity = (special_flatten(model_prune.state_dict())==0).sum()/(special_flatten(model_prune.state_dict())>0).numel() * 100
    print('Retrain Test: {:.2f}, Retrain Validation: {:.2f}, Prune Test: {:.2f}, Prune Validation: {:.2f}, Prune Sparsity: {:.2f}'
          .format(test_accuracy_pre, val_accuracy_pre, test_accuracy_prune, val_accuracy, sparsity))
    
        
    return val_accuracy, sparsity
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', dest = "alpha", help="Gamma Prior Alpha", required=True, type=(float))
    parser.add_argument('--beta', dest = "beta", help="Gamma Prior Beta", required=True, type=(float))
    parser.add_argument('--tau', dest = "tau", help="Tau: Complexity and Error Loss trade-off parameter", required=True, type=(float))
    parser.add_argument('--temp', dest = "temp", help="Temperature: Final softmax temperature for knowledge distillation", required=False, type=(int))
    parser.add_argument('--mixtures', dest = "mixtures", help="Mixtures: Number of Gaussian prior mixtures", required=True, type=(int))
    parser.add_argument('--model', dest = "model", help = "Model to train", required = True, choices = ('SWSModel', 'Lenet_300_100'))
    parser.add_argument('--data', dest = "data", help = "Data to train on - 'full' training data (60k) or 'search' training data(50k)", required = True, choices = ('full','search'))
    parser.add_argument('--savedir', dest = "savedir", help = "Save Directory")
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
        
    retrain_model(alpha, beta, tau, temp, mixtures, model_name, args.data, args.savedir)
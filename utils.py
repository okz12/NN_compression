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


###Training and testing NN
def test_accuracy(data, labels, model):
    model.eval()
    outputs = model(data)
    loss = nn.CrossEntropyLoss()(outputs, labels).data[0]
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.data).sum()
    accuracy = 100.0 * correct/len(labels)
    return accuracy, loss
    
def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
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
def show_weights(model):
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
    for i,params in enumerate(model.parameters()):
        param_list = []
        for pdim in params.size():
            param_list.append(str(pdim))
        if i%2==0:
            dim_str = "x".join(param_list)
        else:
            print (dim_str + " + " + "x".join(param_list))
            
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
    
    
    
def prune_plot(temp, dev_res, perc_res, test_acc_o, train_acc_o, weight_penalty_o, test_acc_kd, train_acc_kd, weight_penalty_kd):
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
def retrain_sws_epoch(model, gmp, optimizer, optimizer_gmp, optimizer_gmp2, criterion, train_loader, tau):
    """
    train model
    
    model: neural network model
    optimizer: optimization algorithm/configuration
    criterion: loss function
    train_loader: training dataset dataloader
    """
    model.train()
    gmp.print_batch = True
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
def show_sws_weights(model, means=0, precisions=0, epoch=-1, accuracy=-1, savefile = ""):
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
    plt.ylabel("Occurrence")
    
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
    plt.ylabel("Occurrence")
    plt.xlim([-1, 1])
    plt.ylim([1e-5, 1e3])
    
    if savefile!="":
        plt.savefig("./figs/{}_{}.png".format(savefile, epoch+1), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
        
###
def draw_sws_graphs(means = -1, stddev = -1, mixprop = -1, acc = -1, savefile=""):
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
        
        
def trueAfterN(ip, N):
    return ((N-1)==ip%N)

def logsumexp(t, w=1, axis=1):
    #print (t.shape)
    t_max, _ = t.max(dim=1)
    if (axis==1):
        t = t-t_max.repeat(t.size(1), 1).t()
    else:
        t = t-t_max.repeat(1, t.size(0)).t()
    t = w * t.exp()
    t = t.sum(dim=axis)
    t.log_()
    return t + t_max
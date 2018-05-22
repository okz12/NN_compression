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
    """
    get model accuracy on a dataset
    
    data: features tensor
    labels: targets tensor
    model: trained model
    """
    model.eval()
    outputs = model(data)
    loss = nn.CrossEntropyLoss()(outputs, labels).data[0]
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.data).sum()
    accuracy = 100.0 * correct/len(labels)
    return accuracy, loss
    
def train_epoch(model, optimizer, criterion, train_loader):
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
        # Forward pass to get output/logits
        outputs = model(images)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels) + 0.001 * get_weight_penalty(model)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
    return model, loss

###
def show_weights(model):
    """
    shows histograms of each parameter layer in model except biases
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
    prints dimensions of a model
    """
    for i,params in enumerate(model.parameters()):
        param_list = []
        for pdim in params.size():
            param_list.append(str(pdim))
        if i%2==0:
            dim_str = "x".join(param_list)
        else:
            print (dim_str + " + " + "x".join(param_list))
            
def get_weight_penalty(model):
    """
    get L2 for weights in each layer in the model
    """
    layer_list = [x.replace(".weight","") for x in model.state_dict().keys() if 'weight' in x]
    wp=0
    for layer in layer_list:
        wp += np.sqrt( ( model.state_dict()[layer + ".weight"].pow(2).sum() + model.state_dict()[layer + ".bias"].pow(2).sum() ) )
    return wp 

###
class model_prune():
    """
    class to help with pruning of layers in a model
    """
    def __init__(self, state_dict):
        """
        model state_dict used to initialise class
        standard deviation and percentile limits for removing 
        """
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
        """
        
        """
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
    """
    plot model pruning graphs
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
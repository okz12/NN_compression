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
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt


###Training and testing NN
def test_accuracy(test_loader,model, kd=False, get_loss=False):
    # Calculate Accuracy         
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        #if use_cuda:
        images = images.cuda()
        images = Variable(images)
        # Forward pass only to get logits/output
        outputs = model.forward(images, kd)
        #CE Loss
        if get_loss:
            loss = loss_criterion(outputs, Variable(labels).cuda()).cpu().data[0]
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += labels.size(0)
        #if use_cuda:
        correct += (predicted.cpu() == labels.cpu()).sum()
        #else:
        #    correct += (predicted == labels).sum()

    accuracy = 100.0 * correct / total
    if get_loss:
        return accuracy, loss
    return accuracy
    
def train_epoch(model, optimizer, criterion, train_loader, kd=False):
    for i, (images, labels) in enumerate(train_loader):

        #if(use_cuda):
        images=images.cuda()
        labels=labels.cuda()
        images = Variable(images)
        labels = Variable(labels)
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model.forward(images, kd)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
    return model, loss

###
def show_weights(model):
    weight_list = ['fc1.weight', 'fc2.weight', 'fc3.weight']
    plt.clf()
    plt.figure(figsize=(18, 3))
    for i,weight in enumerate(weight_list):
        plt.subplot(131 + i)
        fc_w = model.state_dict()[weight]
        sns.distplot(fc_w.view(-1).cpu().numpy())
    plt.show()
    
def print_dims(model):
    for i,params in enumerate(model.parameters()):
        param_list = []
        for pdim in params.size():
            param_list.append(str(pdim))
        if i%2==0:
            dim_str = "x".join(param_list)
        else:
            print dim_str + " + " + "x".join(param_list)

###
class layer_utils():
    def __init__(self, weight):
        self.weight = weight
        self.org_weight = weight.clone()
        self.num_weights = weight.size()[0] * weight.size()[1]
        
        weight_np = np.abs((weight.clone().cpu().numpy()))
        weight_np = weight_np.reshape(-1)
        percentile_limits = [x for x in range (0,101)]
        self.percentile_values = np.percentile(weight_np, percentile_limits)
        
        self.num_pruned = 0
        
    def prune(self, percentile):
        self.weight = self.org_weight.clone()
        zero_idx = self.weight.abs()<self.percentile_values[percentile]
        self.num_pruned = zero_idx.sum()
        self.weight[zero_idx] = 0
        return self.weight
    
    def plot(self):
        plt.clf()
        sns.distplot(self.weight.clone().view(-1).cpu().numpy())
        plt.show()
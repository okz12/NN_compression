import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

class SWSModel(nn.Module):
    def __init__(self):
        super(SWSModel, self).__init__()
        
        self.name = 'SWSModel'
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(1250, 500) 
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500,10)
        self.sm1 = nn.Softmax()
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Convolution 2 
        out = self.cnn2(out)
        #print "{} {},{},{},{}".format("cnn2", out.size(0), out.size(1), out.size(2), out.size(3))
        out = self.relu2(out)
        # Max pool 2 
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        #print "{} {},{}".format("rs", out.size(0), out.size(1))
        # Linear function (readout)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.sm1(out)
        
        return out
    
    
    
class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        
        self.name = 'LeNet_300_100'
        
        self.fc1 = nn.Linear(28*28, 300) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300,100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100,10)
        self.sm1 = nn.Softmax()
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sm1(out)
        
        return out
    
def test_accuracy(test_loader,model):
    # Calculate Accuracy         
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        #if use_cuda:
        images=images.cuda()
        images = Variable(images)
        # Forward pass only to get logits/output
        outputs = model(images)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += labels.size(0)
        #if use_cuda:
        correct += (predicted.cpu() == labels.cpu()).sum()
        #else:
        #    correct += (predicted == labels).sum()

    accuracy = 100.0 * correct / total
    return accuracy
    
def train_epoch(model, optimizer, criterion, train_loader):
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
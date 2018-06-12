import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

### SWS model replicates the model in the soft-weight sharing paper
class SWSModel(nn.Module):
    def __init__(self):
        super(SWSModel, self).__init__()
        
        self.name = 'SWSModel'
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(1250, 500) 
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500,10)
        self.sm1 = nn.Softmax(dim=1)
    
    def forward(self, x):
        #print (x.shape)
        #x = x.view(-1, 1, 1, 28 * 28)
        # Convolution 1
        out = self.conv1(x)
        out = self.relu1(out)
        # Convolution 2 
        out = self.conv2(out)
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
    
    def kd_targets(self, x, T=1.0):
        # Convolution 1
        out = self.conv1(x)
        out = self.relu1(out)
        # Convolution 2 
        out = self.conv2(out)
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
        out = out / T
        out = self.sm1(out)
        return out
    
    
### LeNet 300 - 100 has 2 hidden layers with 300 nodes and 100 nodes
class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        
        self.name = 'LeNet_300_100'
        
        self.fc1 = nn.Linear(28*28, 300) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300,100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100,10)
        self.sm1 = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sm1(out)
        return out
    
    def kd_targets(self, x, T=1.0):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = out / T
        out = self.sm1(out)
        return out
    
class LeNet_300_100_kd(nn.Module):
    def __init__(self):
        super(LeNet_300_100_kd, self).__init__()
        
        self.name = 'LeNet_300_100_kd'
        
        self.fc1 = nn.Linear(28*28, 300) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300,100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100,10)
    
    def forward(self, x, kd=False):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
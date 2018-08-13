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
        x = x.view(-1, 1, 28, 28)
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
        layer_out = {}
        x = x.view(-1, 1, 28, 28)
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
        #out = out / T
        out = self.sm1(out)
        return out
        
    def layer_forward(self, x):
        x = x.view(-1, 1, 28, 28)
        d = x.shape[0]
        out1 = self.conv1(x)
        out2 = self.relu1(out1)
        out2 = self.conv2(out2)
        out3 = self.relu2(out2)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.fc1(out3)
        out4 = self.relu3(out3)
        out4 = self.fc2(out4)
        print (x.shape)
        print (out1.shape)
        print (out2.shape)
        print (out3.shape)
        print (out4.shape)
        return torch.cat((out1.view(d, -1), out2.view(d, -1), out3, out4), 1)
        
    def kd_layer_targets(self, x, T=1.0):
        # Convolution 1
        layer_out = {}
        x = x.view(-1, 1, 28, 28)
        out = self.conv1(x)
        layer_out['conv1.out'] = out
        out = self.relu1(out)
        layer_out['conv1.act'] = out
        # Convolution 2 
        out = self.conv2(out)
        layer_out['conv2.out'] = out
        #print "{} {},{},{},{}".format("cnn2", out.size(0), out.size(1), out.size(2), out.size(3))
        out = self.relu2(out)
        layer_out['conv2.act'] = out
        # Max pool 2 
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        #print "{} {},{}".format("rs", out.size(0), out.size(1))
        # Linear function (readout)
        out = self.fc1(out)
        layer_out['fc1.out'] = out
        out = self.relu3(out)
        layer_out['fc1.act'] = out
        out = self.fc2(out)
        layer_out['fc2.out'] = out
        out = self.sm1(out)
        layer_out['fc2.act'] = out
        return layer_out

class SWSModelKD(nn.Module):
    def __init__(self):
        super(SWSModelKD, self).__init__()
        
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
    
    def forward(self, x):
        #print (x.shape)
        x = x.view(-1, 1, 28, 28)
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
        return out
        
        
class SWSModelConv1(nn.Module):
    def __init__(self):
        super(SWSModelConv1, self).__init__()
        self.name = 'SWSModelConv1'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=2, padding=0)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.conv1(x)
        return out
        
class SWSModelConv2(nn.Module):
    def __init__(self):
        super(SWSModelConv2, self).__init__()
        self.name = 'SWSModelConv2'
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        out = self.conv2(x)
        return out
        
class SWSModelFC1(nn.Module):
    def __init__(self):
        super(SWSModelFC1, self).__init__()
        self.name = 'SWSModelFC1'
        self.fc1 = nn.Linear(1250, 500) 
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out
        
class SWSModelFC2(nn.Module):
    def __init__(self):
        super(SWSModelFC2, self).__init__()
        self.name = 'SWSModelFC2'
        self.fc2 = nn.Linear(500,10)
    
    def forward(self, x):
        out = self.fc2(x)
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
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

    def layer_forward(self, x):
        x = x.view(-1, 28 * 28)
        out1 = self.fc1(x)
        out2 = self.relu1(out1)
        out2 = self.fc2(out2)
        out3 = self.relu2(out2)
        out3 = self.fc3(out3)
        return torch.cat((out1, out2, out3), 1)
        
    def kd_layer_targets(self, x, T=1.0):        
        x = x.view(-1, 28 * 28)
        layer_out = {}
        out = self.fc1(x)
        layer_out['fc1.out'] = out
        out = self.relu1(out)
        layer_out['fc1.act'] = out
        out = self.fc2(out)
        layer_out['fc2.out'] = out
        out = self.relu2(out)
        layer_out['fc2.act'] = out
        out = self.fc3(out)
        layer_out['fc3.out'] = out
        return layer_out

class LeNet_300_100FC1(nn.Module):
    def __init__(self):
        super(LeNet_300_100FC1, self).__init__()
        
        self.name = 'LeNet_300_100FC1'
        self.fc1 = nn.Linear(28*28, 300)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        return out

class LeNet_300_100FC2(nn.Module):
    def __init__(self):
        super(LeNet_300_100FC2, self).__init__()
        
        self.name = 'LeNet_300_100FC2'
        self.fc2 = nn.Linear(300,100)
    
    def forward(self, x):
        #x = x.view(-1, 300)
        out = self.fc2(x)
        return out

class LeNet_300_100FC3(nn.Module):
    def __init__(self):
        super(LeNet_300_100FC3, self).__init__()
        
        self.name = 'LeNet_300_100FC3'
        self.fc3 = nn.Linear(100,10)
    
    def forward(self, x):
        #x = x.view(-1, 10)
        out = self.fc3(x)
        return out


###LeNet-5

class LeNet5(nn.Module):
    def __init__(self):
        self.name = 'LeNet5'
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
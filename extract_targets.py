import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F

model_dir = "./models/"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', dest = "model", help = "Path to model to extract from", required = True)
args = parser.parse_args()
model_file = args.model

loaded_model = torch.load(model_dir + model_file)
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_data_full = Variable(train_dataset.train_data.float()/255.0).cuda()
layer_targets = loaded_model.kd_layer_targets(train_data_full)

model_name = model_file.split(".")[0]
if not os.path.exists(model_dir + model_name):
    os.makedirs(model_dir + model_name)
    
for layer in layer_targets:
    torch.save(layer_targets[layer], "{}{}/{}.m".format(model_dir, model_name, layer))
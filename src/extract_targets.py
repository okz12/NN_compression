import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
from utils_misc import  model_load_dir
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', dest = "model", help = "Path to model to extract from", required = True)
args = parser.parse_args()
model_file = args.model


train_data_full = Variable(train_data(fetch='data')).cuda()

loaded_model = torch.load(model_load_dir + model_file)
layer_targets = loaded_model.kd_layer_targets(train_data_full)

model_name = model_file.split(".")[0]
if not os.path.exists("{}{}_targets".format(model_load_dir, model_name)):
    os.makedirs("{}{}_targets".format(model_load_dir, model_name))
    
for layer in layer_targets:
    torch.save(layer_targets[layer], "{}{}_targets/{}.m".format(model_load_dir, model_name, layer))
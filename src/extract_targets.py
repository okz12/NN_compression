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


def save_targets(model_file):
    train_data_full = Variable(train_data(fetch='data')).cuda()

    loaded_model = torch.load(model_load_dir + model_file + ".m")
    layer_targets = loaded_model.kd_layer_targets(train_data_full)

    if not os.path.exists("{}{}_targets".format(model_load_dir, model_file)):
        os.makedirs("{}{}_targets".format(model_load_dir, model_file))
        
    for layer in layer_targets:
        torch.save(layer_targets[layer], "{}{}_targets/{}.m".format(model_load_dir, model_file, layer))

def get_targets(model_file, temp):
    if not os.path.exists("{}{}_targets".format(model_load_dir, model_file)):
        save_targets(model_file)
    loaded_model = torch.load(model_load_dir + model_file + ".m")
    target_dict = {}
    layers = list(set([x.replace(".bias","").replace(".weight","") for x in loaded_model.state_dict()]))
    for layer in layers:
        output = torch.load("{}{}_targets/{}.out.m".format(model_load_dir, model_file, layer))
        if (len(output.size())<4):#no temp on conv layers
            output = (nn.Softmax()(output/temp)).data#dim=1
        target_dict[layer] = output.clone()
    return target_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', dest = "model", help = "Path to model to extract from", required = True)
    parser.add_argument('--temp', dest = "temp", help="Temperature: Final softmax temperature for knowledge distillation", required=False, type=(int))
    args = parser.parse_args()
    model_file = args.model
    get_targets(model_file, temp)
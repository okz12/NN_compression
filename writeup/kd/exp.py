import sys
sys.path.insert(0,'../../src/')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
import pickle
import model_archs


from utils_model import test_accuracy, train_epoch, retrain_sws_epoch, model_prune, get_weight_penalty, layer_accuracy
from utils_misc import trueAfterN, logsumexp, root_dir, model_load_dir, get_ab, get_sparsity
from utils_sws import GaussianMixturePrior, special_flatten, KL, compute_responsibilies, merger, sws_prune, sws_prune_l2, sws_prune_copy
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
from extract_targets import get_targets
from retrain_layer import init_retrain_layer
from retrain_model import retrain_model

import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest = "mode", help = "Exp number", required = True, type=int)
args = parser.parse_args()
mode = args.mode

if (mode == 11):
    temp_list = [4]
    tau_list = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    start_processing = False
    for temp in temp_list:
        for tau in tau_list:
            if (temp == 5 and tau == 1e-5): #skip
                print ("skip")
            else:
                model, gmp, res = retrain_model(1000, 0.1, 5000, 1000, tau, temp, 16, "LeNet_300_100", "full", 'CEST', False, "./files")

if (mode == 12):
    temp_list = [5]
    tau_list = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    start_processing = False
    for temp in temp_list:
        for tau in tau_list:
            if (temp == 5 and tau == 1e-5): #skip
                print ("skip")
            else:
                model, gmp, res = retrain_model(1000, 0.1, 5000, 1000, tau, temp, 16, "LeNet_300_100", "full", 'CEST', False, "./files")

if (mode == 13):
    temp_list = [6]
    tau_list = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    start_processing = False
    for temp in temp_list:
        for tau in tau_list:
            if (temp == 5 and tau == 1e-5): #skip
                print ("skip")
            else:
                model, gmp, res = retrain_model(1000, 0.1, 5000, 1000, tau, temp, 16, "LeNet_300_100", "full", 'CEST', False, "./files")

if (mode == 14):
    temp_list = [8]
    tau_list = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    start_processing = False
    for temp in temp_list:
        for tau in tau_list:
            if (temp == 5 and tau == 1e-5): #skip
                print ("skip")
            else:
                model, gmp, res = retrain_model(1000, 0.1, 5000, 1000, tau, temp, 16, "LeNet_300_100", "full", 'CEST', False, "./files")
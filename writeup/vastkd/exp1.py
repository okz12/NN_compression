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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest = "mode", help = "Exp number", required = True, type=int)
args = parser.parse_args()
mode = args.mode

if (mode == 1):
    model, gmp, res = retrain_model(250, 10, 2500, 1250, 1e-6, 5, 16, "LeNet_300_100", "full", 'MSEST', False, "./files")

if (mode == 2):
    vlist = [0.1, 1, 10, 100, 1000]
    zvar = 1250
    for var in vlist:
        fn_text = "_var_{}_zvar_{}".format(var,zvar)
        model, gmp, res = retrain_model(250, var, 2500, zvar, 1e-6, 5, 15, "LeNet_300_100", "full", 'MSEST', False, "./files", fn = fn_text)
        show_sws_weights_log(model = model, means = list(gmp.means.data.clone().cpu()), precisions = list(gmp.gammas.data.clone().cpu()))
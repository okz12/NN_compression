import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F

model_dir = "./models/"
import models
from sws_utils import sws_prune
from model_utils import sws_replace
import pickle


test_dataset = dsets.MNIST(root='./data', train=False , transform=transforms.ToTensor(), download=True)
test_data_full = Variable(test_dataset.test_data.float()/255.0).cuda()
test_labels_full = Variable(test_dataset.test_labels).cuda()

model_name = "SWSModel"
model_file = 'mnist_{}_{}'.format(model_name, 100)
model = torch.load("{}{}.m".format(model_dir, model_file)).cuda()

abt_list = []
for alpha in [2500, 25000, 250000]:
    for beta in [0.1, 1, 10, 100]:
        for tau in [1e-4, 1e-5, 1e-6, 1e-7]:
            abt_list.append((alpha, beta, tau))

print (abt_list)

file_dir = model_dir + "mnist_SWSModel_100/"

for exp in abt_list:
    exp_name = "a{}_b{}_t{}".format(float(exp[0]), float(exp[1]), float(exp[2]))
    conv1 = torch.load(file_dir + "mnist_layer_SWSModelConv1_{}.m".format(exp_name))
    conv2 = torch.load(file_dir + "mnist_layer_SWSModelConv2_{}.m".format(exp_name))
    fc1 = torch.load(file_dir + "mnist_layer_SWSModelFC1_{}.m".format(exp_name))
    fc2 = torch.load(file_dir + "mnist_layer_SWSModelFC2_{}.m".format(exp_name))
    
    with open(file_dir + 'mnist_retrain_SWSModelConv1_{}_gmp.p'.format(exp_name),'rb') as f:
        conv1_gmp = pickle.load(f)
    with open(file_dir + 'mnist_retrain_SWSModelConv2_{}_gmp.p'.format(exp_name),'rb') as f:
        conv2_gmp = pickle.load(f)
    with open(file_dir + 'mnist_retrain_SWSModelFC1_{}_gmp.p'.format(exp_name),'rb') as f:
        fc1_gmp = pickle.load(f)
    with open(file_dir + 'mnist_retrain_SWSModelFC2_{}_gmp.p'.format(exp_name),'rb') as f:
        fc2_gmp = pickle.load(f)
    
    retrain_full_model = sws_replace(model, conv1.state_dict(), conv2.state_dict(), fc1.state_dict(), fc2.state_dict())
    retrain_acc = (test_accuracy(test_data_full, test_labels_full, retrain_full_model))
    prune_full_model = sws_replace(model, sws_prune(conv1, conv1_gmp), sws_prune(conv2, conv2_gmp),  sws_prune(fc1, fc1_gmp), sws_prune(fc2, fc2_gmp))
    prune_acc (test_accuracy(test_data_full, test_labels_full, prune_full_model))

    prune_full_model = sws_replace(model, conv1.state_dict(), sws_prune(conv2, conv2_gmp),  sws_prune(fc1, fc1_gmp), sws_prune(fc2, fc2_gmp))
    minus_conv1_acc = (test_accuracy(test_data_full, test_labels_full, prune_full_model))
    prune_full_model = sws_replace(model, sws_prune(conv1, conv1_gmp), conv2.state_dict(),  sws_prune(fc1, fc1_gmp), sws_prune(fc2, fc2_gmp))
    minus_conv2_acc (test_accuracy(test_data_full, test_labels_full, prune_full_model))
    prune_full_model = sws_replace(model, sws_prune(conv1, conv1_gmp), sws_prune(conv2, conv2_gmp),  fc1.state_dict(), sws_prune(fc2, fc2_gmp))
    minus_fc1_acc (test_accuracy(test_data_full, test_labels_full, prune_full_model))
    prune_full_model = sws_replace(model, sws_prune(conv1, conv1_gmp), sws_prune(conv2, conv2_gmp),  sws_prune(fc1, fc1_gmp), fc2.state_dict())
    minus_fc2_acc (test_accuracy(test_data_full, test_labels_full, prune_full_model))
    
    print (exp_name)
    print ("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(retrain_acc[0], prune_acc[0], minus_conv1_acc[0], minus_conv2_acc[0], minus_fc1_acc[0], minus_fc2_acc[0]))
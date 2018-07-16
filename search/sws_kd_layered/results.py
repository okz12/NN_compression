import sys
sys.path.insert(0,'../../src/')
import os
import argparse
from retrain_model import retrain_model
savedir = os.getcwd() + "/models/"
import copy

import pickle
from mnist_loader import train_data
from utils_sws import sws_prune, compressed_model
from utils_model import test_accuracy, layer_accuracy, sws_replace
import torch
from torch.autograd import Variable
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
from utils_misc import model_load_dir

if __name__=="__main__":
    test_data_full =  Variable(test_data(fetch = "data")).cuda()
    test_labels_full =  Variable(test_data(fetch = "labels")).cuda()
    val_data_full =  Variable(search_validation_data(fetch = "data")).cuda()
    val_labels_full =  Variable(search_validation_data(fetch = "labels")).cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest = "start", help="Start Search", required=True, type=(int))
    parser.add_argument('--end', dest = "end", help="End Search", required=True, type=(int))
    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)
    
    with open("../sobol_search_2.p", "rb") as handle:
        params = pickle.load(handle)
    for i in range (start,end):
        print ("exp:{} mean: {}, var: {}, tau: {}, temp: {}, mixtures: {}".format(i, params['mean'][i], params['var'][i], params['tau'][i], float(params['temp'][i]), int(params['mixtures'][i])))
        mean = float(params['mean'][i])
        var = float(params['var'][i])
        beta = mean/var
        alpha = mean * beta
        
        model_name = "SWSModel"
        model_file = 'mnist_{}_{}_{}'.format(model_name, 100, "search")
        model_orig = torch.load(model_load_dir + model_file + '.m').cuda()

        conv1_exp_name = "SWSModelConv1_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format(alpha, beta, 50, float(params['tau'][i]), int(params['mixtures'][i]), int(params['temp'][i]), 'search')
        conv1_model_file = "./models/mnist_SWSModel_100_searchmnist_retrain_{}".format(conv1_exp_name)
        conv2_exp_name = "SWSModelConv2_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format(alpha, beta, 50, float(params['tau'][i]), int(params['mixtures'][i]), int(params['temp'][i]), 'search')
        conv2_model_file = "./models/mnist_SWSModel_100_searchmnist_retrain_{}".format(conv2_exp_name)
        fc1_exp_name = "SWSModelFC1_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format(alpha, beta, 50, float(params['tau'][i]), int(params['mixtures'][i]), int(params['temp'][i]), 'search')
        fc1_model_file = "./models/mnist_SWSModel_100_searchmnist_retrain_{}".format(fc1_exp_name)
        fc2_exp_name = "SWSModelFC2_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format(alpha, beta, 50, float(params['tau'][i]), int(params['mixtures'][i]), int(params['temp'][i]), 'search')
        fc2_model_file = "./models/mnist_SWSModel_100_searchmnist_retrain_{}".format(fc2_exp_name)
        
        conv1_model = torch.load("{}.m".format(conv1_model_file)).cuda()
        with open("{}_gmp.p".format(conv1_model_file), "rb") as handle:
            conv1_gmp = pickle.load(handle)
        conv2_model = torch.load("{}.m".format(conv2_model_file)).cuda()
        with open("{}_gmp.p".format(conv2_model_file), "rb") as handle:
            conv2_gmp = pickle.load(handle)
        fc1_model = torch.load("{}.m".format(fc1_model_file)).cuda()
        with open("{}_gmp.p".format(fc1_model_file), "rb") as handle:
            fc1_gmp = pickle.load(handle)
        fc2_model = torch.load("{}.m".format(fc2_model_file)).cuda()
        with open("{}_gmp.p".format(fc2_model_file), "rb") as handle:
            fc2_gmp = pickle.load(handle)
            
        conv1_res = layer_accuracy(conv1_model, conv1_gmp, model_orig, val_data_full, val_labels_full)
        conv2_res = layer_accuracy(conv2_model, conv2_gmp, model_orig, val_data_full, val_labels_full)
        fc1_res = layer_accuracy(fc1_model, fc1_gmp, model_orig, val_data_full, val_labels_full)
        fc2_res = layer_accuracy(fc2_model, fc2_gmp, model_orig, val_data_full, val_labels_full)
        
        pruned_model = sws_replace(model_orig, sws_prune(conv1_model, conv1_gmp), sws_prune(conv2_model, conv2_gmp), sws_prune(fc1_model, fc1_gmp), sws_prune(fc2_model, fc2_gmp))
        test_acc = test_accuracy(test_data_full, test_labels_full, pruned_model)[0]
        val_acc = test_accuracy(val_data_full, val_labels_full, pruned_model)[0]
        print ("test: {}, val: {}".format(test_acc, val_acc))
        #cm = compressed_model(pruned_model.state_dict(), [conv1_gmp, conv2_gmp, fc1_gmp, fc2_gmp])
        #cr = cm.get_cr(6)[0]
        #print ("CR: {}".format(cr))
        #sp = (cm.binned_weights == 0).sum() / float(cm.binned_weights.size) * 100.0
        #print ("SP: {}".format(sp))
        if not os.path.exists("results.csv"):
            with open("results.csv", "w") as out_csv:
                out_csv.write("Exp, Mean, Var, Tau, Temp, Mixtures, conv1_val, conv1_sp, conv2_val, conv2_sp, fc1_val, fc1_sp, fc2_val, fc_2sp, Test Acc, Val Acc\n")
                out_csv.write(", ".join([str(x) for x in [i, params['mean'][i], params['var'][i], params['tau'][i], int(params['temp'][i]), int(params['mixtures'][i]), 
                                           conv1_res[1], conv1_res[2], conv2_res[1], conv1_res[2], fc1_res[1], fc1_res[2], fc2_res[1], fc2_res[2], test_acc, val_acc]]) + "\n")
        else:
            with open("results.csv", "a") as out_csv:
                out_csv.write(", ".join([str(x) for x in [i, params['mean'][i], params['var'][i], params['tau'][i], int(params['temp'][i]), int(params['mixtures'][i]), 
                                           conv1_res[1], conv1_res[2], conv2_res[1], conv1_res[2], fc1_res[1], fc1_res[2], fc2_res[1], fc2_res[2], test_acc, val_acc]]) + "\n")
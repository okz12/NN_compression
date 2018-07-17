import sys
sys.path.insert(0,'../../src/')
import os
import argparse
from retrain_model import retrain_model
savedir = os.getcwd() + "/models/"

import pickle
from mnist_loader import train_data
from utils_sws import sws_prune, compressed_model
from utils_model import test_accuracy
import torch
from torch.autograd import Variable
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size

from time import time

if __name__=="__main__":
    test_data_full =  Variable(test_data(fetch = "data")).cuda()
    test_labels_full =  Variable(test_data(fetch = "labels")).cuda()
    val_data_full =  Variable(search_validation_data(fetch = "data")).cuda()
    val_labels_full =  Variable(search_validation_data(fetch = "labels")).cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest = "start", help="Start Search", required=True, type=(int))
    parser.add_argument('--end', dest = "end", help="End Search", required=True, type=(int))
    parser.add_argument('--model', dest = "model", help = "Model to extract results from", required = True, choices = ('SWSModel', 'LeNet_300_100'))
    parser.add_argument('--out', dest = 'out', required = True, type=(str))
    args = parser.parse_args()
    model_name = args.model
    outfile = args.out
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

        exp_name = "{}_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format(model_name, alpha, beta, 50, float(params['tau'][i]), int(params['mixtures'][i]), int(params['temp'][i]), 'search')
        model_file = "./models/mnist_retrain_{}".format(exp_name)
        if not os.path.exists("{}.m".format(model_file)):
            print ("File not found: {}.m".format(model_file))
        else:
            model = torch.load("{}.m".format(model_file)).cuda()
            with open("{}_gmp.p".format(model_file), "rb") as handle:
                gmp = pickle.load(handle)
            model.load_state_dict(sws_prune(model, gmp))
            test_acc = test_accuracy(test_data_full, test_labels_full, model)[0]
            val_acc = test_accuracy(val_data_full, val_labels_full, model)[0]
            cm = compressed_model(model.state_dict(), [gmp])
            cr = cm.get_cr(6)[0]
            sp = (cm.binned_weights == 0).sum() / float(cm.binned_weights.size) * 100.0
            if not os.path.exists(outfile):
                with open(outfile, "w") as out_csv:
                    out_csv.write("Exp, Mean, Var, Tau, Temp, Mixtures, Test Acc, Val Acc, Sparse, CR\n")
                    out_csv.write(", ".join([str(x) for x in [i, params['mean'][i], params['var'][i], params['tau'][i], int(params['temp'][i]), int(params['mixtures'][i]), test_acc, val_acc, sp, cr]]) + "\n")
            else:
                with open(outfile, "a") as out_csv:
                    out_csv.write(", ".join([str(x) for x in [i, params['mean'][i], params['var'][i], params['tau'][i], int(params['temp'][i]), int(params['mixtures'][i]), test_acc, val_acc, sp, cr]]) + "\n")
        

#mnist_retrain_SWSModel_a1.3365233866589001e-06_b0.0004151108285503527_r50_t1.6436915121153314e-07_m6_kdT20_search.m
#mnist_retrain_SWSModel_a86791.37374683257_b8.524404751815132_r50_t2.5311641601652367e-05_m9.0_kdT7_search.m
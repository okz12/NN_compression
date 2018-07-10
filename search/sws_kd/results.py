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

from time import time

if __name__=="__main__":
    test_data_full =  Variable(test_data(fetch = "data")).cuda()
    test_labels_full =  Variable(test_data(fetch = "labels")).cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest = "start", help="Start Search", required=True, type=(int))
    parser.add_argument('--end', dest = "end", help="End Search", required=True, type=(int))
    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)

    with open("../sobol_search.p", "rb") as handle:
        params = pickle.load(handle)
    for i in range (start,end):
        print ("mean: {}, var: {}, tau: {}, temp: {}, mixtures: {}".format(params['mean'][i], params['var'][i], params['tau'][i], float(params['temp'][i]), int(params['mixtures'][i])))
        mean = float(params['mean'][i])
        var = float(params['var'][i])
        beta = mean/var
        alpha = mean * beta

        exp_name = "{}_a{}_b{}_r{}_t{}_m{}_kdT{}_{}".format('SWSModel', alpha, beta, 50, float(params['tau'][i]), int(params['mixtures'][i]), int(params['temp'][i]), 'search')
        model_file = "./models/mnist_retrain_{}".format(exp_name)
        print (model_file)
        if not os.path.exists("{}.m".format(model_file)):
            print ("File not found: {}.m".format(model_file))
        else:
            ms = time()
            model = torch.load("{}.m".format(model_file)).cuda()
            me = time()
            print ("Model load time: {}s".format(me - ms))
            with open("{}_gmp.p".format(model_file), "rb") as handle:
                gmp = pickle.load(handle)
            ge = time()
            print ("GMP load time: {}s".format(ge - me))
            model.load_state_dict(sws_prune(model, gmp))
            pe = time()
            print ("Prune time: {}s".format(pe - ge))
            test_acc = test_accuracy(test_data_full, test_labels_full, model)[0]
            te = time()
            print ("Test time: {}s".format(te - pe))
            cm = compressed_model(model.state_dict(), [gmp])
            cre = time()
            print ("Compress time: {}s".format(cre - te))
            cr = cm.get_cr(6)[0]
            sp = (cm.binned_weights == 0).sum() / float(cm.binned_weights.size)
            print (test_acc, sp, cr)
        

#mnist_retrain_SWSModel_a1.3365233866589001e-06_b0.0004151108285503527_r50_t1.6436915121153314e-07_m6_kdT20_search.m
#mnist_retrain_SWSModel_a86791.37374683257_b8.524404751815132_r50_t2.5311641601652367e-05_m9.0_kdT7_search.m
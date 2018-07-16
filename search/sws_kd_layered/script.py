import sys
sys.path.insert(0,'../../src/')
import os
import argparse
from retrain_layer import init_retrain_layer
savedir = os.getcwd() + "/models/"

import pickle

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest = "start", help="Start Search", required=True, type=(int))
    parser.add_argument('--end', dest = "end", help="End Search", required=True, type=(int))
    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)

    with open("../sobol_search_2.p", "rb") as handle:
        params = pickle.load(handle)
    for i in range (start,end):
        print ("Experiment {}".format(i))
        print ("mean: {}, var: {}, tau: {}, temp: {}, mixtures: {}".format(params['mean'][i], params['var'][i], params['tau'][i], float(params['temp'][i]), int(params['mixtures'][i])))
        mean = float(params['mean'][i])
        var = float(params['var'][i])
        beta = mean/var
        alpha = mean * beta
        for layer in range(1,5):
            print ("Layer: {}".format(layer))
            init_retrain_layer(alpha, beta, float(params['tau'][i]), int(params['mixtures'][i]), float(params['temp'][i]), "search", layer, savedir)
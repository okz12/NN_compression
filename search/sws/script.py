import sys
sys.path.insert(0,'../../src/')
import os
import argparse
from retrain_model import retrain_model
savedir = os.getcwd() + "/models/"

import pickle
import numpy as np

def main(job_id, params):
    print (params)
    mean = float(params['mean'])
    var = float(params['var'])
    beta = mean/var
    alpha = mean * beta
    acc, sp = retrain_model(alpha, beta, float(params['tau']), 0, int(params['mixtures']), 'SWSModel', 'search', savedir)
    acc_score = (100-acc)**2.5
    sp_score = (100-sp)**1.5
    score = acc_score + sp_score
    print ("Final Score: {} Acc Score: {} Sp Score: {}".format(score, acc_score, sp_score))
    print ("=====================================\n")
    return {
        "score"       : score, 
        "min_sparsity" : sp-88, 
        "min_accuracy" : acc-93
    }

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest = "start", help="Start Search", required=True, type=(int))
    parser.add_argument('--end', dest = "end", help="End Search", required=True, type=(int))
    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)

    with open("../sobol_search_2.p", "rb") as handle:
        params = pickle.load(handle)
    '''
    tupled_params = [tuple(row) for row in np.vstack((params['mean'], params['var'], params['tau'], params['mixtures'])).T][:end]
    unique_params = list(set(tupled_params))

    reduced_params = {}
    reduced_params['mean'] = np.array([x[0] for x in unique_params])
    reduced_params['var'] = np.array([x[1] for x in unique_params])
    reduced_params['tau'] = np.array([x[2] for x in unique_params])
    reduced_params['mixtures'] = np.array([int(x[3]) for x in unique_params])
    params = reduced_params

    
    for i in range (start,len(unique_params)):
    '''
    for i in range (start,end):
        print ("Experiment {}".format(i))
        print ("mean: {}, var: {}, tau: {}, temp: {}, mixtures: {}".format(params['mean'][i], params['var'][i], params['tau'][i], int(0), int(params['mixtures'][i])))
        mean = float(params['mean'][i])
        var = float(params['var'][i])
        beta = mean/var
        alpha = mean * beta
        acc, sp = retrain_model(alpha, beta, float(params['tau'][i]), int(0), int(params['mixtures'][i]), 'LeNet_300_100', 'search', savedir, False)
        
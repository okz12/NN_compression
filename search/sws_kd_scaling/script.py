import sys
sys.path.insert(0,'../../src/')
import os
import argparse
from retrain_model import retrain_model
savedir = os.getcwd() + "/models/"

import pickle

def main(job_id, params):
    print (params)
    mean = float(params['mean'])
    var = float(params['var'])
    beta = mean/var
    alpha = mean * beta
    acc, sp = retrain_model(alpha, beta, float(params['tau']), float(params['temp']), int(params['mixtures']), 'SWSModel', 'search', savedir, False)
    acc_score = (100-acc)**2.5
    sp_score = (100-sp)**1.5
    score = acc_score + sp_score
    print ("Final Score: {} Acc Score: {} Sp Score: {}".format(score, acc_score, sp_score))
    print ("=====================================\n")
    if(job_id == -1):
        return score
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
    for i in range (start,end):
        print ("Experiment {}".format(i))
        print ("mean: {}, var: {}, tau: {}, temp: {}, mixtures: {}".format(params['mean'][i], params['var'][i], params['tau'][i], float(params['temp'][i]), int(params['mixtures'][i])))
        mean = float(params['mean'][i])
        var = float(params['var'][i])
        beta = mean/var
        alpha = mean * beta
        acc, sp = retrain_model(alpha, beta, float(params['tau'][i]), float(params['temp'][i]), int(params['mixtures'][i]), 'SWSModel', 'search', savedir, True)
        
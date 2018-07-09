import sys
sys.path.insert(0,'../../src/')
import os
from retrain_model import retrain_model
savedir = os.getcwd() + "/models/"

import pickle

def main(job_id, params):
    print (params)
    mean = float(params['mean'])
    var = float(params['var'])
    beta = mean/var
    alpha = mean * beta
    acc, sp = retrain_model(alpha, beta, float(params['tau']), float(params['temp']), int(params['mixtures']), 'SWSModel', 'search', savedir)
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
    with open("../sobol_search.p", "rb") as handle:
        params = pickle.load(handle)
    print (params)
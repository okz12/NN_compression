import sys
sys.path.insert(0,'../../src/')

from retrain_model import retrain_model
model_dir = "./models/"

def main(job_id, params):
    print (params)
    mean = float(params['mean'])
    var = float(params['var'])
    beta = mean/var
    alpha = mean * beta
    acc, sp = retrain_model(alpha, beta, float(params['tau']), 0, int(params['mixtures']), 'SWSModel', 'search')
    acc_score = (100-acc)**2.5
    sp_score = (100-sp)**2.5
    score = acc_score + sp_score
    print ("Final Score: {} Acc Score: {} Sp Score: {}".format(score, acc_score, sp_score))
    print ("=====================================\n")
    return {
        "score"       : score, 
        "min_sparsity" : sp-88, 
        "min_accuracy" : acc-93
    }
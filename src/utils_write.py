import numpy as np
import pickle

def loadfile(mean, var, zmean, zvar, tau, temp, mixtures, model, data_size = "full",  scaling = False, model_save_dir = "", fn="", file = "res", dset="mnist"):
    s = "s" if scaling else "f" 
    exp_name = "{}_m{}_zm{}_r{}_t{}_m{}_kdT{}_{}_{}".format(model, mean, zmean, 50, tau, int(mixtures), int(temp), s, data_size) + fn
    if (file=='res'):
        with open(model_save_dir + '/{}_retrain_res_{}.p'.format(dset, exp_name),'rb') as f:
            file = pickle.load(f)
    if (file=='gmp'):
        with open(model_save_dir + '/{}_retrain_gmp_{}.p'.format(dset, exp_name),'rb') as f:
            file = pickle.load(f)
    if (file=='model'):
        file = torch.load(model_save_dir + '/{}_retrain_model_{}.m'.format(dset, exp_name))
    return file

def stack_weights(weight_dict):
    weights = np.array([], dtype=np.float32)
    for layer in weight_dict:
        weights = np.hstack( (weights, weight_dict[layer] ) )
    return weights

#Codebook
#Create Indexbook
def create_index(bpi, nz_list, weight_p):
    max_skip = 2**bpi
    if (nz_list[0] != 0):
        ridx_list = [0]
        w_list = [0]
    else:
        ridx_list = []
        w_list = []

    cidx = 0
    for nz in nz_list:

        nzc = nz - cidx
        ins_0 = int(np.floor(nzc/max_skip))
        nidx = int(nzc - (max_skip * ins_0))
        for i in range (ins_0):
            w_list.append(0)
            ridx_list.append(max_skip)
        w_list.append(1)
        ridx_list.append(nidx)
        #print (nz, cidx, nzc)
        cidx = nz
    lri = len(weight_p) - sum(ridx_list)
    if (lri>0):
        ridx_list.append(lri-1)
        w_list.append(0)
    return ridx_list, w_list

def recover_index(ridx_list, w_list, sz):
    rec = np.zeros(sz)
    for r,w in zip (list(np.array(ridx_list).cumsum()), w_list):
        rec[r] = w
    return rec

def cr_calc(res):
    mixtures = res['mixtures']
    weight_p = stack_weights(res['prune_weights'])
    weight_s = (weight_p != 0)
    nz_list = list(np.where(weight_s == True)[0])
    cb = 32.0 * mixtures
    sb = 32.0 * (len(res['scale'][-1]) - 1)
    orig_net = len(weight_p) * 32.0
    
    res_dict = {}
    
    for bpi in range(2,10):
        rl, wl = create_index (bpi, nz_list, weight_p)
        #rec = recover_index(rl, wl, len(weight_p))
        ib = bpi * len(rl)
        bpw = np.ceil(np.log2(res['mixtures']))
        wb = bpw * len(wl)
        cr = orig_net / (cb + wb + ib + sb)
        res2_dict = {}
        res2_dict['cr'] = cr
        res2_dict['on'] = orig_net
        res2_dict['wb'] = wb
        res2_dict['ib'] = ib
        res2_dict['sb'] = sb
        res2_dict['cb'] = cb
        res_dict[bpi] = res2_dict
    return res_dict
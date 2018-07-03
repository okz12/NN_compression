import numpy as np

root_dir = "/home/okz21/NNC/NN_compression/"
model_load_dir = root_dir + "models/"

def trueAfterN(ip, N):
    return ((N-1)==ip%N)

def logsumexp(t, w=1, axis=1):
    #print (t.shape)
    t_max, _ = t.max(dim=1)
    if (axis==1):
        t = t-t_max.repeat(t.size(1), 1).t()
    else:
        t = t-t_max.repeat(1, t.size(0)).t()
    t = w * t.exp()
    t = t.sum(dim=axis)
    t.log_()
    return t + t_max

def nplogsumexp(ns):
    max_val = np.max(ns)
    ds = ns - max_val
    sumOfExp = np.exp(ds).sum()
    return max_val + np.log(sumOfExp)
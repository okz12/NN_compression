import numpy as np

import os
cwd = os.getcwd()
if "okz" in cwd:
	root_dir = "/home/okz21/NNC/NN_compression/"
else:
	root_dir = "/root/NN_compression/"
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


def get_sparsity(model_prune):
	sp_zeroes = 0
	sp_elem = 0
	for layer in model_prune.state_dict():
		sp_zeroes += float((model_prune.state_dict()[layer].view(-1) == 0).sum())
		sp_elem += float(model_prune.state_dict()[layer].view(-1).numel())
	sp = sp_zeroes/sp_elem * 100.0
	return sp
	
def get_ab(mean, var):
	beta = mean/var
	alpha = mean * beta
	return (alpha, beta)
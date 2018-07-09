'''
Create 10000 quasi-random low-discrepancy points in a dictionary for 
1. gamma prior mean (range: 0.001 - 100000, logarithmically spaced)
2. gamma prior variance (range: 0.01 - 1000, logarithmically spaced)
3. trade-off parameter for between accuracy and clustering (range: 1e-8 - 1e-3, logarithmically spaced)
4. knowledge distillation temperature (range: 2 - 20)
5. number of GMM mixtures (range: 3 - 16)
Dictionary is saved in the search folder
'''
from SALib.sample import saltelli
import numpy as np
import pickle

problem = {
  'num_vars': 5,
  'names': ['mean', 'var', 'tau', 'temp', 'mixtures'],
  'bounds': [[np.log(0.001), np.log(100000)],
    [np.log(0.01), np.log(1000)],
    [np.log(1e-8), np.log(1e-3)],
    [1.5,20.49],
    [2.5,16.49]]
}

# Generate samples
param_values = saltelli.sample(problem, 10000)

params = {}
params['mean'] = np.exp(param_values[:,0])
params['var'] = np.exp(param_values[:,1])
params['tau'] = np.exp(param_values[:,2])
params['temp'] = np.around(param_values[:,3])
params['mixtures'] = np.around(param_values[:,4])

with open('../search/sobol_search.p', 'wb') as handle:
    pickle.dump(params, handle)
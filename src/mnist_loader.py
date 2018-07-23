'''
Provides separate functions to load MNIST data
'''
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

batch_size = 128

def load_data(begin, end, train, fetch, onehot = False):
    dataset = dsets.MNIST(root='/home/okz21/NNC/NN_compression/data/mnist/', train=train , transform=transforms.ToTensor(), download=True)
    if(train):
        data = (dataset.train_data.float()/255.0)
        labels = (dataset.train_labels)
    else:
        data = (dataset.test_data.float()/255.0)
        labels = (dataset.test_labels)

    if (onehot):
        one_hot =  torch.zeros(labels.shape[0], 10)
        one_hot.scatter_(1, labels.view(-1,1),1)
        labels = one_hot

    if(fetch == 'labels'):
        return labels[begin:end]
    elif(fetch == 'data'):
        return data[begin:end]
    else:
        return torch.utils.data.TensorDataset(data[begin:end], labels[begin:end])

#Initial training data 0-50K
def search_train_data(begin=0, end=50000, fetch='dataset', onehot = False):
    return load_data(begin, end, True, fetch, onehot)

#Hyperparameter search data 40-50K (reduced to speed up optimization)
def search_retrain_data(begin=40000, end=50000, fetch='dataset', onehot = False):
    return load_data(begin, end, True, fetch, onehot)

#Hyperparameter validation data 50-60k (used for bayesian optimization)
def search_validation_data(begin=50000, end=60000, fetch='dataset', onehot = False):
    return load_data(begin, end, True, fetch, onehot)

#Full training data - used for training and retraining after hyperparameter search
def train_data(begin=0, end=60000, fetch='dataset', onehot = False):
    return load_data(begin, end, True, fetch, onehot)

#Full test data - used for accuracy testing
def test_data(begin=0, end=10000, fetch='dataset', onehot = False):
    return load_data(begin, end, False, fetch, onehot)
    
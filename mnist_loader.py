'''
Provides separate functions to load MNIST data
'''
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

batch_size = 128

def load_data(begin, end, train, fetch):
    dataset = dsets.MNIST(root='./data', train=train , transform=transforms.ToTensor(), download=True)
    if(train):
        data = (dataset.train_data.float()/255.0)
        labels = (dataset.train_labels)
    else:
        data = (dataset.test_data.float()/255.0)
        labels = (dataset.test_labels)
    if(fetch == 'labels'):
        return labels[begin:end]
    elif(fetch == 'data'):
        return data[begin:end]
    else:
        return torch.utils.data.TensorDataset(data[begin:end], labels[begin:end])

#Initial training data 0-50K
def search_train_data(begin=0, end=50000, fetch='dataset'):
    return load_data(begin, end, True, fetch)

#Hyperparameter search data 40-50K (reduced to speed up optimization)
def search_retrain_data(begin=40000, end=50000, fetch='dataset'):
    return load_data(begin, end, True, fetch)

#Hyperparameter validation data 50-60k (used for bayesian optimization)
def search_validation_data(begin=50000, end=60000, fetch='dataset'):
    return load_data(begin, end, True, fetch)

#Full training data - used for training and retraining after hyperparameter search
def train_data(begin=0, end=60000, fetch='dataset'):
    return load_data(begin, end, True, fetch)

#Full test data - used for accuracy testing
def test_data(begin=0, end=10000, fetch='dataset'):
    return load_data(begin, end, False, fetch)
    
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import IPython.display as ipd
import imageio
#import os
#model_dir = "/".join(os.getcwd().split("/")[:-1] + ['models/'])
model_dir = "./models/"
import models
from utils import show_sws_weights, test_accuracy, train_epoch, retrain_sws_epoch, show_weights, model_prune, print_dims, get_weight_penalty, prune_plot, draw_sws_graphs, trueAfterN, logsumexp
import copy
from tensorboardX import SummaryWriter
import pickle

writeTensorboard = False
if(writeTensorboard):
    writer = SummaryWriter('tensorboard/run1/')

import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
sns.set(color_codes=True)
sns.set_style("whitegrid")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', dest = "alpha", help="Gamma Prior Alpha", required=True)
parser.add_argument('--beta', dest = "beta", help="Gamma Prior Beta", required=True)
args = parser.parse_args()
alpha = float(args.alpha)
beta = float(args.beta)


#Data
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False , transform=transforms.ToTensor(), download=True)

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
train_data_full = Variable(train_dataset.train_data.float()/255.0).cuda()
test_data_full = Variable(test_dataset.test_data.float()/255.0).cuda()
train_labels_full = Variable(train_dataset.train_labels).cuda()
test_labels_full = Variable(test_dataset.test_labels).cuda()
use_cuda = torch.cuda.is_available()

#arguments
model_name = 'LeNet_300_100'

class GaussianMixturePrior(Module):
    def __init__(self, nb_components, network_weights, pi_zero, init_var = 0.25, zero_ab = (5e3,2), ab = (2.5e4,1), **kwargs):
        super(GaussianMixturePrior, self).__init__()
        
        self.nb_components = nb_components 
        self.network_weights = [p.view(-1) for p in network_weights]
        self.pi_zero = pi_zero
        
        #Build
        J = self.nb_components
        pi_zero = self.pi_zero
        
        #    ... means
        init_means = torch.linspace(-0.6, 0.6, J - 1)
        self.means = Variable(init_means.cuda(), requires_grad=True)
        
        #precision
        init_stds = torch.FloatTensor(np.tile(init_var, J) )
        self.gammas = Variable( (- torch.log(torch.pow(init_stds, 2))).cuda(), requires_grad=True)
        
        #mixing proportions
        init_mixing_proportions = torch.ones((J - 1))
        init_mixing_proportions *= (1. - pi_zero) / (J - 1)
        self.rhos = Variable((init_mixing_proportions).cuda(), requires_grad=True)
        self.print_batch=True
        
        self.zero_ab = zero_ab
        self.ab = ab
        print ("0-component Mean: {} Variance: {}".format(zero_ab[0]/zero_ab[1], zero_ab[0]/(zero_ab[1]**2)))
        print ("Non-zero component Mean: {} Variance: {}".format(ab[0]/ab[1], ab[0]/(ab[1]**2)))
        #self.loss = Variable(torch.cuda.FloatTensor([0.]), requires_grad=True)
        
    def call(self, mask=None):
        J=self.nb_components
        loss = Variable(torch.cuda.FloatTensor([0.]), requires_grad=True)
        means = torch.cat(( Variable(torch.cuda.FloatTensor([0.]), requires_grad=True) , self.means), 0)
        #mean=self.means
        precision = self.gammas.exp()
        
        min_rho = self.rhos.min()
        mixing_proportions = (self.rhos - min_rho).exp()
        mixing_proportions = (1 - self.pi_zero) * mixing_proportions/mixing_proportions.sum()
        mixing_proportions = torch.pow(mixing_proportions, 2)
        mixing_proportions = torch.cat(( Variable(torch.cuda.FloatTensor([self.pi_zero])) , mixing_proportions), 0)
        
        for weights in self.network_weights:
            weight_loss = self.compute_loss(weights, mixing_proportions, means, precision)
            if(gmp.print_batch):
                print ("Layer Loss: {:.3f}".format(float(weight_loss.data)))
            loss = loss + weight_loss
        
        
        # GAMMA PRIOR ON PRECISION
        # ... for the zero component
        #Replacing gather with indexing -- same calculation?
        (alpha, beta) = self.zero_ab
        #print (torch.gather(self.gammas, 0, Variable(torch.cuda.LongTensor([0,1,2]))))
        neglogprop = (1 - alpha) * self.gammas[0] + beta * precision[0]
        if(gmp.print_batch):
            print ("0-neglogprop Loss: {:.3f}".format(float(neglogprop.data)))
        loss = loss + neglogprop.sum()
        # ... and all other component
        alpha, beta = self.ab
        neglogprop = (1 - alpha) * self.gammas[1:J] + beta * precision[1:J]
        if(gmp.print_batch):
            print ("Remaining-neglogprop Loss: {:.3f}".format(float(neglogprop.sum().data)))
        loss = loss + neglogprop.sum()
        gmp.print_batch=False
        return loss
        
        
    def compute_loss(self, weights, mixing_proportions, means, precision):
        diff = weights.expand(means.size(0), -1) - means.expand(weights.size(0), -1).t()
        unnormalized_log_likelihood = (-(diff ** 2)/2).t() * precision
        #unnormalized_log_likelihood = (-1/2) * precision.matmul((diff ** 2))
        Z = precision.sqrt() / (2 * np.pi)
        #global myt
        #myt=unnormalized_log_likelihood
        log_likelihood = logsumexp(unnormalized_log_likelihood, w=(mixing_proportions * Z), axis=1)
        return -log_likelihood.sum()
        
#model = models.LeNet_300_100().cuda()
#print_dims(model)
model = torch.load(model_dir + 'mnist_{}_{}.m'.format(model_name, 100)).cuda()
gmp = GaussianMixturePrior(16, [x for x in model.parameters()], 0.99, ab = (alpha, beta))

test_acc = test_accuracy(test_data_full, test_labels_full, model)
train_acc = test_accuracy(train_data_full, train_labels_full, model)
acc_history = np.array([train_acc[0], test_acc[0]])
stddev_history = np.sqrt(1. / gmp.gammas.exp().data.clone().cpu().numpy())
mean_history = gmp.means.data.clone().cpu().numpy()
mixprop_history = gmp.rhos.exp().data.clone().cpu().numpy()

sws_param1 = [gmp.means]
sws_param2 = [gmp.gammas, gmp.rhos]
#ipd.display(ipd.Markdown("**Default Training**"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_gmp = torch.optim.Adam(sws_param1, lr=1e-4)
optimizer_gmp2 = torch.optim.Adam(sws_param2, lr=3e-3)
graph_title='original_model/'

retraining_epochs = 200
decay = np.linspace(5e-7, 5e-6, retraining_epochs)
#decay = 5e-7 * np.power(10, decay)
#decay = 5e-5 * (1 - 1 / np.power(10, decay))

exp_name = "a{}_b{}_r{}".format(alpha, beta, retraining_epochs)

for epoch in range(retraining_epochs):
    print("Epoch: {}".format(epoch+1))
    #tau = float(decay[epoch])
    tau=5e-7
    model, loss = retrain_sws_epoch(model, gmp, optimizer, optimizer_gmp, optimizer_gmp2, criterion, train_loader, tau)

    test_acc = test_accuracy(test_data_full, test_labels_full, model)
    train_acc = test_accuracy(train_data_full, train_labels_full, model)
    weight_penalty = get_weight_penalty(model)

    stddev_history = np.vstack((stddev_history,  np.sqrt(1. / gmp.gammas.exp().data.clone().cpu().numpy()) ))
    mean_history = np.vstack((mean_history, gmp.means.data.clone().cpu().numpy() ))
    mixprop_history = np.vstack((mixprop_history, gmp.rhos.exp().data.clone().cpu().numpy() ))
    acc_history = np.vstack(( acc_history, np.array([train_acc[0], test_acc[0]]) ))

    if(writeTensorboard):
        writer.add_scalars(graph_title + 'CrossEntropyLoss', {'Test': test_acc[1], 'Train': train_acc[1]}, epoch+1)
        writer.add_scalars(graph_title + 'Accuracy', {'Test': test_acc[0], 'Train': train_acc[0]}, epoch+1)
        writer.add_scalars(graph_title + 'L2', {'L2' : weight_penalty}, epoch+1)
        for name, param in model.named_parameters():
            writer.add_histogram(graph_title + name, param.clone().cpu().data.numpy(), epoch+1, bins='doane')
    if (trueAfterN(epoch, 10)):
        gmp.print_batch = True
        print ('Tau:{}'.format(tau))
        print('Epoch: {}. Training Accuracy: {:.2f}. Test Accuracy: {}'.format(epoch+1, train_acc[0], test_acc[0]))
        print ( "Means: {}".format(list(np.around(gmp.means.data.clone().cpu().numpy(),3))) )
        print ( "Mixing Proportions: {}".format(list(np.around(gmp.rhos.data.clone().cpu().numpy(),3))) )
        print ( "Precisions: {}".format(list(np.around(gmp.gammas.data.clone().cpu().numpy(),3))) )
    show_sws_weights(model = model, means = list(gmp.means.data.clone().cpu()), precisions = list(gmp.gammas.data.clone().cpu()), epoch = epoch, accuracy = test_acc[0], savefile = exp_name)
        #show_all_weights(model)

draw_sws_graphs(mean_history, stddev_history, mixprop_history, acc_history, exp_name)
images = []
filenames = ["figs/{}_{}.png".format(exp_name, x) for x in range(1,retraining_epochs+1)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./exp/{}.gif'.format(exp_name), images)
torch.save(model, model_dir + 'mnist_retrain_m{}_a{}_b{}_r{}.m'.format(model.name, alpha, beta, retraining_epochs))
with open(model_dir + 'mnist_retrain_m{}_a{}_b{}_r{}_gmp.p'.format(model.name, alpha, beta, retraining_epochs),'wb') as f:
    pickle.dump(gmp, f)
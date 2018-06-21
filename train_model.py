import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F

model_dir = "./models/"
import models
from model_utils import test_accuracy, train_epoch
from helpers import trueAfterN
from plot_utils import print_dims

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest = "model", help = "Model to train", required = True)
parser.add_argument('--epochs', dest = "epochs", help = "Number of training epochs", required = True)
args = parser.parse_args()
model_name = args.model
training_epochs = int(args.epochs)


#Data
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False , transform=transforms.ToTensor(), download=True)

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_full = Variable(test_dataset.test_data.float()/255.0).cuda()
test_labels_full = Variable(test_dataset.test_labels).cuda()

if model_name == "SWSModel":
	model = models.SWSModel().cuda()
	#test_data_full = test_data_full.view(10000, 1, 28, 28)
else:
	model = models.LeNet_300_100().cuda()
	
print ("Model Name: {} Epochs: {}".format(model.name, training_epochs))
print_dims(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.000)

for epoch in range(training_epochs):
	model, loss = train_epoch(model, optimizer, criterion, train_loader)

	if (trueAfterN(epoch, 10)):
		test_acc = test_accuracy(test_data_full, test_labels_full, model)
		print('Epoch: {}. Test Accuracy: {:.2f}'.format(epoch+1, test_acc[0]))
   
torch.save(model, model_dir + 'mnist_{}_{}.m'.format(model.name, training_epochs))
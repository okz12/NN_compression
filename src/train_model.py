#python train_model.py --model SWSModel --epochs 100 --data search
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable

import model_archs
from utils_model import test_accuracy, train_epoch
from utils_misc import trueAfterN, model_load_dir
from utils_plot import print_dims
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
import argparse

def train_model(model_name, data_size, training_epochs):
	if(data_size == 'search'):
		train_dataset = search_train_data()
	if(data_size == 'full'):
		train_dataset = train_data()
	
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_data_full = Variable(test_data(fetch='data')).cuda()
	test_labels_full = Variable(test_data(fetch='labels')).cuda()

	if model_name == "SWSModel":
		model = model_archs.SWSModel().cuda()
	else:
		model = model_archs.LeNet_300_100().cuda()

	print ("Model Name: {} Epochs: {} Data: {}".format(model.name, training_epochs, data_size))
	print_dims(model)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.000)

	for epoch in range(training_epochs):
		model, loss = train_epoch(model, optimizer, criterion, train_loader)

		if (trueAfterN(epoch, 10)):
			test_acc = test_accuracy(test_data_full, test_labels_full, model)
			print('Epoch: {}. Test Accuracy: {:.2f}'.format(epoch+1, test_acc[0]))

	torch.save(model, model_load_dir + 'mnist_{}_{}_{}.m'.format(model.name, training_epochs, data_size))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', dest = "model", help = "Model to train", required = True, choices = ('SWSModel', 'LeNet_300_100'))
	parser.add_argument('--epochs', dest = "epochs", help = "Number of training epochs", required = True, type=int)
	parser.add_argument('--data', dest = "data", help = "Data to train on - 'full' training data (60k) or 'search' training data(50k)", required = True, choices = ('full','search'))
	args = parser.parse_args()
	model_name = args.model
	training_epochs = int(args.epochs)
	train_model(model_name, args.data, training_epochs)
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
from utils_misc import  model_load_dir
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
import model_archs
import argparse


def save_targets(model_file):
	train_data_full = Variable(train_data(fetch='data')).cuda()

	loaded_model = torch.load(model_load_dir + model_file + ".m")
	layer_targets = loaded_model.kd_layer_targets(train_data_full)

	if not os.path.exists("{}{}_targets".format(model_load_dir, model_file)):
		os.makedirs("{}{}_targets".format(model_load_dir, model_file))
		
	for layer in layer_targets:
		torch.save(layer_targets[layer], "{}{}_targets/{}.m".format(model_load_dir, model_file, layer))

def get_targets(model_file, temp = 0, layers=[]):
	if not os.path.exists("{}{}_targets".format(model_load_dir, model_file)):
		save_targets(model_file)
	loaded_model = torch.load(model_load_dir + model_file + ".m")
	target_dict = {}
	if layers == []:
		layers = list(set([x.replace(".bias",".out").replace(".weight",".out") for x in loaded_model.state_dict()]))
	for layer in layers:
		output = torch.load("{}{}_targets/{}.m".format(model_load_dir, model_file, layer))
		if (len(output.size())<4 and temp != 0):#no temp on conv layers
			output = (nn.Softmax(dim=1)(output/temp))
		target_dict[layer] = output.clone()
	return target_dict

def get_layer_data(target_dir, temp, layer, model_name, data_size, loss_type = 'MSEHNA'):
	x_start = 0
	x_end = 60000
	if (data_size == "search"):
		x_start = 40000
		x_end = 50000
	if ("SWSModel" in model_name):
		if (layer == 1):
			layer_model = model_archs.SWSModelConv1().cuda()
			input = Variable(train_data(fetch = "data")[x_start:x_end]).cuda()
			output = get_targets(target_dir, temp, ["conv1.out"])["conv1.out"][x_start:x_end]
		if (layer == 2):
			layer_model = model_archs.SWSModelConv2().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["conv1.out"])["conv1.out"][x_start:x_end])
			output = (get_targets(target_dir, temp, ["conv2.out"])["conv2.out"][x_start:x_end])
		if (layer == 3):
			layer_model = model_archs.SWSModelFC1().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["conv2.out"])["conv2.out"][x_start:x_end])
			output = get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end]
		if (layer == 4):
			layer_model = model_archs.SWSModelFC2().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end])
			output = get_targets(target_dir, temp, ["fc2.out"])["fc2.out"][x_start:x_end]

	if ("LeNet_300_100" in model_name):
		if (layer == 1):
			layer_model = model_archs.LeNet_300_100FC1().cuda()
			input = Variable(train_data(fetch = "data")[x_start:x_end]).cuda()
			output = get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end]
		if (layer == 2):
			layer_model = model_archs.LeNet_300_100FC2().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["fc1.out"])["fc1.out"][x_start:x_end])
			output = (get_targets(target_dir, temp, ["fc2.out"])["fc2.out"][x_start:x_end])
		if (layer == 3):
			layer_model = model_archs.LeNet_300_100FC3().cuda()
			input = nn.ReLU()(get_targets(target_dir, temp, ["fc2.out"])["fc2.out"][x_start:x_end])
			output = get_targets(target_dir, temp, ["fc3.out"])["fc3.out"][x_start:x_end]

	if (loss_type == 'MSEHA' or loss_type =='CESH'):
		output = nn.ReLU()(output)
	if (loss_type == 'CESH'):
		output = nn.Softmax(dim=1)(output/temp)

	dataset = torch.utils.data.TensorDataset(input.data, output.data)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	return layer_model, loader

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_file', dest = "model", help = "Path to model to extract from", required = True)
	parser.add_argument('--temp', dest = "temp", help="Temperature: Final softmax temperature for knowledge distillation", required=False, type=(int))
	args = parser.parse_args()
	model_file = args.model
	save_targets(model_file)
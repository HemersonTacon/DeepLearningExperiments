import argparse
import time
import os
import platform
import random
import datetime as dt
import numpy as np

from automatize_helper import  save_infos
from simpleModel import print_best_acc, plot_and_save
from applications_train import load_dataset, app_model, create_obs, fine_tuning_train
from sklearn.model_selection import ParameterGrid

def _get_Args():

	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory containing the dataset splited in train, validation and test folders")
	parser.add_argument("params", help="Text file with list of parameters to find best parameters set")
	parser.add_argument("-p", "--percentage", help = "Percentage of all combinations to test", type=float, default=0.01)

	return parser.parse_args()

def get_outdir():

	# check OS to save in right place
	if platform.system() == 'Windows':
		return "G:\\Meu Drive\\Mestrado\\Experimentos Titan\\UCF101"
	else:
		return "Experimentos/UCF101"


def load_params(params_file):

	with open(params_file, 'r') as f:
		params = {line.split()[0]:tuple(line.split()[1:] if line.split()[0] == 'net_model' else map(float, line.split()[1:]))  for line in f.readlines()}

	return params

def set_args(params, epochs):

	args = _get_Args()
	params['dir'] = args.dir
	params['epochs'] = epochs
	params['optimizer'] = 'SGD'

	return params

def train(params, model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, epochs=500):

	if not 'lambdal1' in params.keys():
		params['lambdal1'] = 0
	if not 'lambdal2' in params.keys():
		params['lambdal2'] = 0


	start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
	s = time.time()
	score, hist, name_weights, name_weights_best = fine_tuning_train(params['net_model'], model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, params['learning_rate'], params['batch_size'], epochs, True, params['lambdal1'], params['lambdal2'])
	end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
	t = time.time()
	time_formated = str(dt.timedelta(seconds=t-s))

	names = plot_and_save(hist, name_weights, False)
	idx = print_best_acc(hist)
	outdir = get_outdir()
	obs = create_obs(params['net_model'], params['img_size'], params['dense'], params['dropout'], False, True, True)

	args = set_args(params, epochs)
	save_infos(os.path.basename(__file__), args, 'ImageNet', hist, idx, score, name_weights, name_weights_best, *names, [start, end, time_formated], outdir, obs, use_app='grid_ft')

	res = {
		'acc': hist.history['val_acc'][idx],
		'loss': hist.history['val_loss'][idx]
	}

	return res

def grid_search_serial(params, data_dir, metric='acc', maximize=True, _type='random', grid_downsample=1):

	# load dataset
	num_classes, img_size, train_gen, valid_gen, test_gen, num_train, num_valid, num_test = load_dataset(3, data_dir, params['net_model'][0])
	#num_classes2, img_size2, train_gen2, valid_gen2, test_gen2, num_train2, num_valid2, num_test2 = load_dataset(3, data_dir, 'resnet50')

	# ParameterGrid values
	grid_params = ParameterGrid(params)
	# Convert object ParameterGrid to List
	list_params = list(grid_params)
	num_comb_to_test = int(len(list_params)*grid_downsample)

	print("Testing {} combinations".format(num_comb_to_test))

	results = []
	infos = []
	params_tested = []

	for i in range(num_comb_to_test):

		print("RUNNING PROCCESS {} FROM {}".format(i+1, num_comb_to_test))

		# Parameters order to do the search
		if(_type == 'linear'):
		    _choice = i
		elif(_type == 'reverse'):
		    _choice = len(list_params) - 1
		elif(_type == 'random'):
		    _choice = random.randrange(len(list_params))
		else:
			raise ValueError("Invalid type. Must choose one of {}".format(['linear', 'reverse','random']))

		_params = list_params.pop(_choice)
		params_tested.append(_params)

		# make the model according to the picked parameters
		# don't need the base model here
		_, model = app_model(img_size, 3, _params['net_model'], _params['dense'], _params['dropout'], num_classes)

		# incorporate image size into parameters dictionary
		_params['img_size'] = img_size
		# train with the selected parameters
		res = train(_params, model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, epochs=500)

		infos.append(res)
		results.append(res[metric])


	if(maximize):
		idx = np.argmax(results)
	else:
		idx = np.argmin(results)

	best_params = params_tested[idx]
	best_infos = infos[idx]

	return best_params, best_infos

def _main(args):

	# from txt load parameters and put them into a Dictionary
	params = load_params(args.params)

	# do grid search
	best_params, best_infos = grid_search_serial(params, args.dir, grid_downsample = args.percentage)
	print("\n\n **** Best parameters found: {}".format(best_params))
	print("\n **** Best accuracy found: {}".format(best_infos))

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()

	# print info about starting time
	start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
	print("\n*********\nBegin time: {}\n*********\n".format(start))
	# set some variables to record time elapsed during execution
	s = time.time()
	# run
	_main(args)
	t = time.time()
	# print info about starting time
	end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
	print("\n*********\nEnd time: {}\n*********\n".format(end))
	# print total elapsed time
	time_formated = str(dt.timedelta(seconds=t-s))
	print("\n########\nTotal Elapsed time: {}\n########\n".format(time_formated))

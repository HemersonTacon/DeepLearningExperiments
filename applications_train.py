#Keras import
from keras.applications import InceptionV3, DenseNet121, Xception, ResNet50, InceptionResNetV2
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.models import Model

# my scripts import
from simpleModel import get_dirs, formatTime, plot_and_save, print_best_acc, handle_opt_params
from automatize_helper import save_infos

# python imports
import os
import numpy as np
import argparse
import time
import datetime as dt
import platform

def _get_Args():

	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory containing the dataset splited in train, validation and test folders")
	parser.add_argument("net_model", help="Network model", choices=['inceptionv3','densenet', 'xception', 'resnet50', 'inceptionresnetv2', 'squeezenet'])
	parser.add_argument("-d", "--dense", help = "Model with extra dense layers", type=int, nargs='+')
	parser.add_argument("-do", "--dropout", help = "Dropout in all dense layers", type=float, nargs='+')
	parser.add_argument("-tl", "--transferlearning", help="Do transfer learning phase", action="store_true")
	parser.add_argument("-ft", "--finetuning", help="Do finetuning phase", action="store_true")
	parser.add_argument("-all", help = "Fine tunne all the layers", action="store_true")
	parser.add_argument("-opt","--optimizer", help = "Optimization algorithm", default='sgd', choices=['adam', 'sgd', 'rmsprop'])
	parser.add_argument("-opt_params","--optimizer_parameters", help = "Optimizer parameters", nargs='*')
	parser.add_argument("-e","--epochs", help = "Number of times to run the entire dataset", type=int, default=5)
	parser.add_argument("-lr","--learning_rate", help = "Learning rate", type=float, default=1e-3)
	parser.add_argument("-bs","--batch_size", help = "Number of samples presented by iteration", type=int , default=16)
	#parser.add_argument("-l","--log", help = "Log file", default='log_transfer_learning')
	parser.add_argument("-rm","--resume_model", help = "Name of model to load")


	return parser.parse_args()

def get_img_size(network):

	sizes = {'inceptionv3':299, 'densenet':224, 'xception':299, 'resnet50':224, 'inceptionresnetv2':299, 'squeezenet':224}

	return sizes[network]

def get_network(network):

	sizes = {'inceptionv3':InceptionV3, 'densenet':DenseNet121, 'xception':Xception, 'resnet50':ResNet50, 'inceptionresnetv2':InceptionResNetV2, 'squeezenet':None}

	return sizes[network]
def my_schedule(total_epoch):
	def schedule(epoch, lr):
		# diminui o alfa em 10 vezes quando chega nas epocas 100 e 200
		if epoch == int(total_epoch*0.4) or epoch == int(total_epoch*0.8):
			lr = lr/10.0
		return lr
	return schedule

def transfer_learning_train(net_name, base_model, model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, lr, bs, eps):

	# first: train only the top layers (which were randomly initialized)
	for layer in base_model.layers:
		layer.trainable = False

	# compile the model
	model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	print('Model loaded')

	# callback to save the progress on csv and save best weights on h5 file
	csv_logger = CSVLogger('log_transfer_learning.csv', append=True, separator=';')
	checkpoint_path = 'transfer_learning_{}_best_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	best_acc = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	# training
	print("Transfer learning")
	hist = model.fit_generator(train_gen, steps_per_epoch=num_train//bs, epochs=eps, callbacks=[csv_logger, best_acc], validation_data=valid_gen, validation_steps=num_valid//bs, shuffle=True)

	if test_gen:
		score = simple_model.evaluate_generator(test_gen, steps=num_test//bs)
	else:
		score = "No set to test"

	print(score)

	model_name = 'transfer_learning_{}_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	model.save_weights(model_name)

	return model, score, hist, model_name, checkpoint_path

def fine_tuning_train(net_name, model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, lr, bs, eps, all=False):

	#TODO: implement parameter to allow train in just some modules of models
	#layers_to_keep_freeze = {'inceptionv3':249, 'densenet':249, 'xception':249, 'resnet50':249, 'inceptionresnetv2':249, 'squeezenet':249}

	if all:
		for layer in model.layers:
			layer.trainable = True
	'''else:
		for layer in model.layers[:layers_to_keep_freeze[net_name]]:
			layer.trainable = False
		for layer in model.layers[layers_to_keep_freeze[net_name]:]:
			layer.trainable = True'''

	model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

	# callback to save the progress on csv and save best weights on h5 file
	csv_logger = CSVLogger('log_transfer_learning.csv', append=True, separator=';')
	checkpoint_path = 'fine_tuning_{}_best_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	best_acc = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	lrs = LearningRateScheduler(my_schedule(eps), verbose=0)

	hist = model.fit_generator(train_gen, steps_per_epoch=num_train//bs, epochs=eps, callbacks=[csv_logger, best_acc], validation_data=valid_gen, validation_steps=num_valid//bs, shuffle=True)

	if test_gen:
		score = simple_model.evaluate_generator(test_gen, steps=num_test//bs)
	else:
		score = "No set to test"

	print(score)

	model_name = 'fine_tuning_{}_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	model.save_weights(model_name)

	return score, hist, model_name, checkpoint_path

def train(indir, net_model, dense, dpout, tl, ft, lr, bs, eps, rm, all, nb_channel = 3):

	# dictionaries to save informations about executions
	tl_infos, ft_infos = {}, {}

	###############################################
	############## Preparing Dataset ##############
	###############################################

	num_channels = nb_channel

	# loading dataset and getting the samples amount of each set
	dir_train, dir_valid, dir_test = get_dirs(indir)

	num_classes = len(os.listdir(dir_train))

	num_train = sum([len(files) for r, d, files in os.walk(dir_train)])
	num_test = sum([len(files) for r, d, files in os.walk(dir_test)])
	num_valid = sum([len(files) for r, d, files in os.walk(dir_valid)])

	# retrieving the image size according to network
	img_size = get_img_size(net_model)

	# preparing the generators for each set
	train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
	train_gen = train_datagen.flow_from_directory(dir_train, target_size = (img_size, img_size), batch_size = bs, class_mode = 'categorical')

	try:
		test_datagen = ImageDataGenerator(rescale=1./255)
		test_gen = test_datagen.flow_from_directory(dir_test, target_size = (img_size, img_size), batch_size = bs, class_mode = 'categorical')
	except:
		test_gen = None

	valid_datagen = ImageDataGenerator(rescale=1./255)
	valid_gen = valid_datagen.flow_from_directory(dir_valid, target_size = (img_size, img_size), batch_size = bs, class_mode = 'categorical')

	print('Dataset loaded.')

	###############################################
	############### Preparing Model ###############
	###############################################

	# getting the base model
	network = get_network(net_model)
	base_model = network(include_top=False, weights='imagenet', input_shape=(img_size, img_size, num_channels))
	X = base_model.output
	X = Flatten()(X)

	if dense and dpout:
		count = 0
		for units, rate in zip(dense, dpout):

			X = Dense(units, use_bias=False, name='extra_dense{}'.format(count))(X)
			X = BatchNormalization(name='extra_bn{}'.format(count))(X)
			X = Activation('relu', name='extra_activation{}'.format(count))(X)
			X = Dropout(rate, name='extra_dropout{}'.format(count))(X)

	# adding a top layer and setting the final model
	predictions = Dense(num_classes, activation='softmax')(X)
	model = Model(inputs = base_model.input, output=predictions)

	# load weights from a previous train
	if rm:
		model.load_weights(rm)

	###############################################
	############### Training phases ###############
	###############################################

	#transfer learning phase
	if tl:
		start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		s = time.time()
		model, score, hist, name_weights, name_weights_best = transfer_learning_train(net_model, base_model, model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, lr, bs, eps)
		end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		t = time.time()
		time_formated = str(dt.timedelta(seconds=t-s))

		names = plot_and_save(hist, name_weights, False)
		idx = print_best_acc(hist)

		tl_infos = {'hist':hist, 'idx': idx, 'score': score, 'weights': [name_weights, name_weights_best], 'plots': names, 'time': [start, end, time_formated]}

	#fine tuning phase
	if ft:
		start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		s = time.time()
		score, hist, name_weights, name_weights_best = fine_tuning_train(net_model, model, train_gen, test_gen, valid_gen, num_train, num_valid, num_test, lr, bs, eps, all)
		t = time.time()
		end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		time_formated = str(dt.timedelta(seconds=t-s))

		names = plot_and_save(hist, name_weights, False)
		idx = print_best_acc(hist)

		ft_infos = {'hist':hist, 'idx': idx, 'score': score, 'weights': [name_weights, name_weights_best], 'plots': names, 'time': [start, end, time_formated]}

	return tl_infos, ft_infos, img_size

def _main(args):

	return train(args.dir, args.net_model, args.dense, args.dropout, args.transferlearning, args.finetuning, args.learning_rate, args.batch_size, args.epochs, args.resume_model, args.all)

def create_obs(net_model, img_size, dense, dropout, transferlearning, finetuning, all_layers):

	obs = "Usada {} com entrada {}".format(net_model, img_size)
	if  dense:
		obs += " e {} camadas densas {}".format(len(dense), dense)
	if dropout:
		obs += " e dropout {}".format( dropout)
	obs += "\nRealizada etapa de"
	if transferlearning:
		obs += " transfer learning"
	if transferlearning and  finetuning:
		obs += " e"
	if finetuning:
		obs += " fine tunning"
	if all_layers:
		obs += " com todas camadas"

	return obs

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()

	# print info about starting time
	start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
	print("\n*********\nBegin time: {}\n*********\n".format(start))
	# set some variables to record time elapsed during execution
	s = time.time()
	# run training and return data about execution
	tl_infos, ft_infos, img_size = _main(args)
	t = time.time()
	# print info about starting time
	end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
	print("\n*********\nEnd time: {}\n*********\n".format(end))
	# print total elapsed time
	time_formated = str(dt.timedelta(seconds=t-s))
	print("\n########\nTotal Elapsed time: {}\n########\n".format(time_formated))

	# check OS to save in right place
	if platform.system() == 'Windows':
		outdir = "G:\\Meu Drive\\Mestrado\\Experimentos Titan\\UCF101"
	else:
		outdir = "Experimentos/UCF101"

	obs = create_obs(args.net_model, img_size, args.dense, args.dropout, args.transferlearning, args.finetuning, args.all)

	if args.transferlearning:
		save_infos(os.path.basename(__file__), args, args.resume_model if args.resume_model else 'ImageNet', tl_infos['hist'], tl_infos['idx'], tl_infos['score'], tl_infos['weights'][0], tl_infos['weights'][1], *tl_infos['plots'], tl_infos['time'], outdir, obs, use_app='tl')
	if args.finetuning:
		save_infos(os.path.basename(__file__), args, args.resume_model if args.resume_model else 'ImageNet', ft_infos['hist'], ft_infos['idx'], ft_infos['score'], ft_infos['weights'][0], ft_infos['weights'][1], *ft_infos['plots'], ft_infos['time'], outdir, obs, use_app='ft')

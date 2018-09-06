#Keras import
from keras.applications import InceptionV3, DenseNet121, Xception, ResNet50, InceptionResNetV2
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation, AveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.regularizers import l1_l2
from pprint import pprint
from keras import __version__ as keras_version

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
import re
from packaging import version

def da_regex(s, pat=re.compile(r"(hf|vf|zoom_0.(\d+))")):
	if not pat.match(s):
		msg = "{} argument didn't pass in regex test:\n\n\t -> {}".format(s, pat)
		raise argparse.ArgumentTypeError(msg)
	return s

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
	parser.add_argument("-l1","--lambda1", help = "L1 kernel regularization", type=float)
	parser.add_argument("-l2","--lambda2", help = "L2 kernel regularization", type=float)
	#parser.add_argument("-l","--log", help = "Log file", default='log_transfer_learning')
	parser.add_argument("-rm","--resume_model", help = "Name of model to load")
	parser.add_argument("-da","--data_augmentation", help = "Kind of data augmentation used", nargs='+', type=da_regex)
	parser.add_argument("-c","--center", help = "Apply featurewise center on samples", action="store_true")
	parser.add_argument("-std_norm","--stdev_normalization", help = "Apply featurewise standard normalization on samples", action="store_true")

	#TODO: usar aumento de dados de acordo com a passagem de argumentos

	return parser.parse_args()

def get_img_size(network):

	sizes = {'inceptionv3':299, 'densenet':224, 'xception':299, 'resnet50':224,
				'inceptionresnetv2':299, 'squeezenet':224}

	return sizes[network]

def get_network(network):

	sizes = {'inceptionv3':InceptionV3, 'densenet':DenseNet121,
			 'xception':Xception, 'resnet50':ResNet50,
			 'inceptionresnetv2':InceptionResNetV2, 'squeezenet':None}

	return sizes[network]

def my_schedule(total_epoch):
	def schedule(epoch, lr):
		# diminui o alfa em 10 vezes quando chega nas epocas 100 e 200
		if epoch == int(total_epoch*1./3.) or epoch == int(total_epoch*2./3.):
			lr = lr/10.0
		return lr
	return schedule

def transfer_learning_train(net_name, base_model, model, train_gen, test_gen,
						valid_gen, num_train, num_valid, num_test, lr, bs, eps):

	# first: train only the top layers (which were randomly initialized)
	for layer in base_model.layers:
		layer.trainable = False

	# compile the model
	model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy',
														metrics=['accuracy'])

	print('Model loaded')

	# callback to save the progress on csv and save best weights on h5 file
	csv_logger = CSVLogger('log_transfer_learning.csv', append=True, separator=';')
	checkpoint_path = 'transfer_learning_{}_best_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	best_acc = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0,
			save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early_stopper = EarlyStopping(monitor='val_acc', patience=30)

	# training
	print("Transfer learning")
	hist = model.fit_generator(train_gen, steps_per_epoch=num_train//bs,
								epochs=eps, callbacks=[best_acc, early_stopper],
								validation_data=valid_gen,
								validation_steps=num_valid//bs, shuffle=True)

	if test_gen:
		score = simple_model.evaluate_generator(test_gen, steps=num_test//bs)
	else:
		score = "No set to test"

	print(score)

	model_name = 'transfer_learning_{}_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	model.save_weights(model_name)

	K.clear_session()

	return model, score, hist, model_name, checkpoint_path

def set_kernel_reg(model, lambdal1 = 0, lambdal2 = 0):
	"""
	 Apply kernel regularization to keras model

	 Args:
		 model: Instance of `Model` not compiled yet
		 lambda1 (float): L1 regularization factor
		 lambda2 (float): L2 regularization factor

	 Returns:
		 Return the same model but with kernel_regularizer configured
	 """


	for layer in model.layers:
		if hasattr(layer, 'kernel_regularizer'):
			layer.kernel_regularizer = l1_l2(l1 = lambdal1, l2 = lambdal2)

	return model

def fine_tuning_train(net_name, model, train_gen, test_gen, valid_gen,
						num_train, num_valid, num_test, lr, bs, eps, all=False,
						lambdal1=0, lambdal2=0):

	#TODO: implement parameter to allow train in just some modules of models
	#layers_to_keep_freeze = {'inceptionv3':249, 'densenet':249, 'xception':249, 'resnet50':249, 'inceptionresnetv2':249, 'squeezenet':249}

	if type(bs) == float:
		bs = int(bs)

	if all:
		for layer in model.layers:
			layer.trainable = True
	'''else:
		for layer in model.layers[:layers_to_keep_freeze[net_name]]:
			layer.trainable = False
		for layer in model.layers[layers_to_keep_freeze[net_name]:]:
			layer.trainable = True'''
	# set kernel regularization
	if lambdal1 or lambdal2:
		model = set_kernel_reg(model, lambdal1, lambdal2)

	model.compile(optimizer=SGD(lr=lr, momentum=0.9),
					loss='categorical_crossentropy', metrics=['accuracy'])

	# callback to save the progress on csv and save best weights on h5 file
	csv_logger = CSVLogger('log_transfer_learning.csv', append=True, separator=';')
	checkpoint_path = 'fine_tuning_{}_best_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	best_acc = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0,
			save_best_only=True, save_weights_only=False, mode='auto', period=1)
	lrs = LearningRateScheduler(my_schedule(eps), verbose=0)
	# Helper: Stop when we stop learning.
	early_stopper = EarlyStopping(monitor='val_acc', patience=25)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
									patience=10, min_lr=1e-6)

	hist = model.fit_generator(train_gen, steps_per_epoch=num_train//bs,
								epochs=eps, callbacks=[best_acc, lrs],
								validation_data=valid_gen,
								validation_steps=num_valid//bs, shuffle=True)

	if test_gen:
		score = simple_model.evaluate_generator(test_gen, steps=num_test//bs)
	else:
		score = "No set to test"

	print(score)

	model_name = 'fine_tuning_{}_lrate{}_bsize{}_epochs{}_{}.h5'.format(net_name, lr, bs, eps, time.time())
	model.save_weights(model_name)

	K.clear_session()

	return score, hist, model_name, checkpoint_path

# Subsample mean and stdev based on https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
def sample_statistics(sample_list, actual_mean=0, actual_std=0, sample_count=1):

	# take pixels mean and stddev for each channel
	mean = sample_list.mean((0,1,2))
	std = sample_list.std((0,1,2))

	samples, rows, cols, _ = sample_list.shape
	total_samples = (samples*rows*cols)

	if sample_count == 1:
		return mean, std, total_samples
	else:
		m = (sample_count*actual_mean + total_samples*mean)/(sample_count+total_samples)
		s = np.sqrt((sample_count*actual_std**2 + total_samples*std**2 +
					 sample_count*(actual_mean-m)**2 + total_samples*(mean-m)**2)
					 /(sample_count+total_samples))

		return m, s, sample_count+total_samples

# Hacking featurewise_center adn featurewise_std_normalization to work with flow_from_directory
# Based on code: https://github.com/smileservices/keras_utils/blob/master/utils.py#L57
def get_img_fit_flow(image_config, fit_sample_size, flow_dir_config):
	"""
	Sample the generators to get fit data

	Args:
		image_config (dict): Holds the vars for data augmentation
		fit_sample_size (float): Subunit multiplier to get the sample size for normalization
		flow_dir_config (dict): Holds the vars for flow_from_directory

	Returns:
		A `DirectoryIterator` yielding tuples of `(x, y)`
					where `x` is a numpy array containing a batch
					of images with shape `(batch_size, *target_size, channels)`
					and `y` is a numpy array of corresponding labels.

	"""

	if (('featurewise_std_normalization' in image_config and
		image_config['featurewise_std_normalization']) or
		('featurewise_center' in image_config and
		image_config['featurewise_center'])):

		mean = np.zeros((1, 1, 3))
		stdev = np.zeros((1, 1, 3))
		count_samples = 1

		img_gen = ImageDataGenerator()
		batches = img_gen.flow_from_directory(**flow_dir_config)

		batch_size = flow_dir_config['batch_size'] if 'batch_size' in flow_dir_config else 16

		for i in range(batches.samples//batch_size):
			imgs, labels = next(batches)
			idx = np.random.choice(imgs.shape[0], int(batch_size*fit_sample_size),
																replace=False)
			mean, stdev, count_samples = sample_statistics(imgs[idx], mean,
														stdev, count_samples)


	new_img_gen = ImageDataGenerator(**image_config)
	if 'featurewise_std_normalization' in image_config and image_config['featurewise_std_normalization']:
		new_img_gen.std = stdev
	if 'featurewise_center' in image_config and image_config['featurewise_center']:
		new_img_gen.mean = mean


	print("\n\n ****Settings:\n")
	pprint(vars(new_img_gen))

	# unpack the necessary ones
	return new_img_gen.flow_from_directory(**flow_dir_config), stdev, mean

def load_dataset(bs, indir, net_model, center = True,
					std_norm = True, data_aug = {}):

	###############################################
	############## Preparing Dataset ##############
	###############################################

	# loading dataset and getting the samples amount of each set
	dir_train, dir_valid, dir_test = get_dirs(indir)

	num_classes = len(os.listdir(dir_train))

	num_train = sum([len(files) for r, d, files in os.walk(dir_train)])
	num_test = sum([len(files) for r, d, files in os.walk(dir_test)])
	num_valid = sum([len(files) for r, d, files in os.walk(dir_valid)])

	# retrieving the image size according to network
	img_size = get_img_size(net_model)

	# preparing the generators for each set
	if center or std_norm:
		data_aug['featurewise_center'] = center
		data_aug['featurewise_std_normalization'] = std_norm
		flow_dir = {'directory': dir_train, 'target_size': (img_size, img_size),
		 			'batch_size': bs, 'class_mode': 'categorical'}
		train_gen, stdev, mean = get_img_fit_flow(data_aug, 1, flow_dir)
	else:
		data_aug['rescale'] = 1./255
		train_datagen = ImageDataGenerator(**data_aug)
		train_gen = train_datagen.flow_from_directory(dir_train,
					target_size = (img_size, img_size), batch_size = bs,
					class_mode = 'categorical')



	if center or std_norm:
		valid_datagen = ImageDataGenerator(featurewise_center = True,
						featurewise_std_normalization = True)
		valid_datagen.mean = mean
		if std_norm:
			valid_datagen.std = stdev
	else:
		valid_datagen = ImageDataGenerator(rescale=1./255)

	valid_gen = valid_datagen.flow_from_directory(dir_valid,
				target_size = (img_size, img_size), batch_size = bs,
				class_mode = 'categorical')

	try:
		if center or std_norm:
			test_datagen = ImageDataGenerator(featurewise_center = True,
							featurewise_std_normalization = True)
			test_datagen.mean = mean
			if std_norm:
				test_datagen.std = stdev
		else:
			test_gen = ImageDataGenerator(rescale=1./255)

		test_gen = test_datagen.flow_from_directory(dir_test,
					target_size = (img_size, img_size), batch_size = bs,
					class_mode = 'categorical')

	except:
		test_gen = None

	print('Dataset loaded')

	return (num_classes, img_size, train_gen, valid_gen, test_gen,
			num_train, num_valid, num_test)

def app_model(img_size, num_channels, net_model, dense,
			  dpout, num_classes, rm=None):

	###############################################
	############### Preparing Model ###############
	###############################################

	# getting the base model
	network = get_network(net_model)
	base_model = network(include_top=False, weights='imagenet',
						 input_shape=(img_size, img_size, num_channels))
	X = base_model.output
	# Since version 2.2.0 ResNet50 do not include the last Average Polling layer when include_top parameter is False
	# but we need it
	if (net_model == 'resnet50' and version.parse(keras_version) >= version.parse('2.2.0')):
		X = AveragePooling2D((7, 7))(X)
	X = Flatten()(X)

	if dense and dpout:
		count = 0
		# assure the right type will be passed
		if not type(dense) == list:
			dense = list(map(int,[dense]))
		if not type(dpout) == list:
			dpout = list(map(int,[dpout]))

		for units, rate in zip(dense, dpout):

			X = Dense(units, use_bias=False, name='extra_dense{}'.format(count))(X)
			X = BatchNormalization(name='extra_bn{}'.format(count))(X)
			X = Activation('relu', name='extra_activation{}'.format(count))(X)
			X = Dropout(rate, name='extra_dropout{}'.format(count))(X)
			count+=1

	# adding a top layer and setting the final model
	predictions = Dense(num_classes, activation='softmax')(X)
	model = Model(inputs = base_model.input, output=predictions)

	# load weights from a previous train
	if rm:
		model.load_weights(rm)

	return base_model, model

def train(indir, net_model, dense, dpout, tl, ft, lr, bs, eps, rm, all,
		  nb_channel=3, l1=0, l2=0, center = True, std_norm = True,
		  data_aug = {}):

	# dictionaries to save informations about executions
	tl_infos, ft_infos = {}, {}

	(num_classes, img_size, train_gen, valid_gen, test_gen,
	num_train, num_valid, num_test) = load_dataset(bs, indir, net_model,
													center = center,
													std_norm = std_norm,
													data_aug = data_aug)

	base_model, model = app_model(img_size, nb_channel, net_model,
									dense, dpout, num_classes, rm)

	###############################################
	############### Training phases ###############
	###############################################

	#transfer learning phase
	if tl:
		start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		s = time.time()

		(model, score, hist, name_weights,
		name_weights_best) = transfer_learning_train(net_model, base_model,
							model, train_gen, test_gen, valid_gen, num_train,
							num_valid, num_test, lr, bs, eps)

		end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		t = time.time()
		time_formated = str(dt.timedelta(seconds=t-s))

		names = plot_and_save(hist, name_weights, False)
		idx = print_best_acc(hist)

		tl_infos = {'hist': hist, 'idx': idx, 'score': score,
					'weights': [name_weights, name_weights_best],
					'plots': names, 'time': [start, end, time_formated]}

	#fine tuning phase
	if ft:
		start = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		s = time.time()
		(score, hist, name_weights,
		name_weights_best) = fine_tuning_train(net_model, model, train_gen,
								test_gen, valid_gen, num_train, num_valid,
								num_test, lr, bs, eps, all, l1, l2)

		t = time.time()
		end = time.strftime("%d/%b/%Y %H:%M:%S", time.localtime())
		time_formated = str(dt.timedelta(seconds=t-s))

		names = plot_and_save(hist, name_weights, False)
		idx = print_best_acc(hist)

		ft_infos = {'hist': hist, 'idx': idx, 'score': score,
					'weights': [name_weights, name_weights_best],
					'plots': names, 'time': [start, end, time_formated]}

	return tl_infos, ft_infos, img_size

def retrieve_data_aug(data_aug_list):

	data_aug = {}

	if data_aug_list:
		pat=re.compile(r"zoom_0.(\d+)")
		data_aug = {'zoom_range':float('0.'+pat.match(item).group(1))
						for item in data_aug_list if pat.match(item)}
		if 'hf' in data_aug_list:
			data_aug['horizontal_flip'] = True
		if 'vf' in data_aug_list:
			data_aug['vertical_flip'] = True

	return data_aug

def _main(args):

	return train(args.dir, args.net_model, args.dense, args.dropout,
				 args.transferlearning, args.finetuning, args.learning_rate,
				 args.batch_size, args.epochs, args.resume_model,args.all,
				 l1=args.lambda1, l2=args.lambda2, center = args.center,
				 std_norm = args.stdev_normalization,
				 data_aug = retrieve_data_aug(args.data_augmentation))

def create_obs(net_model, img_size, dense, dropout, transferlearning,
				finetuning, all_layers):

	if not type(dense)==list:
		dense = [dense]

	obs = "Usada {} com entrada {}".format(net_model, img_size)
	if	dense:
		obs += " e {} camadas densas {}".format(len(dense), dense)
	if dropout:
		obs += " e dropout {}".format( dropout)
	obs += "\nRealizada etapa de"
	if transferlearning:
		obs += " transfer learning"
	if transferlearning and finetuning:
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

	obs = create_obs(args.net_model, img_size, args.dense, args.dropout,
						args.transferlearning, args.finetuning, args.all)

	if args.transferlearning:
		save_infos(os.path.basename(__file__), args,
					args.resume_model if args.resume_model else 'ImageNet',
					tl_infos['hist'], tl_infos['idx'], tl_infos['score'],
					tl_infos['weights'][0], tl_infos['weights'][1],
					*tl_infos['plots'], tl_infos['time'], outdir, obs,
					use_app='tl')

	if args.finetuning:
		save_infos(os.path.basename(__file__), args,
					args.resume_model if args.resume_model else 'ImageNet',
					ft_infos['hist'], ft_infos['idx'], ft_infos['score'],
					ft_infos['weights'][0], ft_infos['weights'][1],
					*ft_infos['plots'], ft_infos['time'], outdir, obs,
					use_app='ft')

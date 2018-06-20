#Keras import
from keras import applications, optimizers, regularizers
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Input
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from transform import valid_generator, train_generator
from finetunninginception import transferWeights
from automatize_helper import save_infos

import os
import numpy as np
import argparse
import time
import sys
import time
import cv2
from math import sqrt

# tamanho antes do crop
img_size_for_crop = 192
# tamanho da entrada da rede
img_size = 129
num_channels = 3

# configuracoes da rede para a parte convolucional

CFG_FEATURES = {
	'A': ({"type": "Conv2d", "filters": 32, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 128, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 256, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 256, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 512, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'B': ({"type": "Conv2d", "filters": 64, "size":(7,7), "stride": 2, "padding": "valid"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 128, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 256, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 256, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 512, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'C': ({"type": "Conv2d", "filters": 32, "size":5, "stride": 2, "padding": "valid"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":5, "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 128, "size":5, "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 256, "size":5, "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'D': ({"type": "Conv2d", "filters": 32, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 128, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 256, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'E': ({"type": "Conv2d", "filters": 32, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'F': ({"type": "Conv2d", "filters": 32, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2}
		 ),
	'G': ({"type": "Conv2d", "filters": 32, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 32, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 128, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 256, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 256, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'H': ({"type": "Conv2d", "filters": 16, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 16, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 16, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 32, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 128, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 )
}

# configuracoes da rede para a parte densa
CFG_CLASSIFIER = {
	'A': ({"type": "Linear", "out_features": 1024},
		  {"type": "Dropout", "rate": 0.8},
		  {"type": "Linear", "out_features": 2048},
		  {"type": "Dropout", "rate": 0.8}
		 ),
	'B': ({"type": "Linear", "out_features": 512},
		  {"type": "Dropout", "rate": 0.7},
		  {"type": "Linear", "out_features": 1024},
		  {"type": "Dropout", "rate": 0.7}
		 ),
	'C': ({"type": "Linear", "out_features": 256},
		  {"type": "Dropout", "rate": 0.6},
		  {"type": "Linear", "out_features": 512},
		  {"type": "Dropout", "rate": 0.6}
		 ),
	'D': ({"type": "Linear", "out_features": 128},
		  {"type": "Dropout", "rate": 0.5},
		  {"type": "Linear", "out_features": 256},
		  {"type": "Dropout", "rate": 0.5}
		 ),
	'E': ({"type": "Linear", "out_features": 64},
		  {"type": "Dropout", "rate": 0.2},
		  {"type": "Linear", "out_features": 128},
		  {"type": "Dropout", "rate": 0.2}
		 ),
	'F': ({"type": "Linear", "out_features": 64},
		  {"type": "Dropout", "rate": 0.7}
		 ),
	'G': ({"type": "Linear", "out_features": 128},
		  {"type": "Dropout", "rate": 0.7}
		 ),
	'H': ({"type": "Linear", "out_features": 256},
		  {"type": "Dropout", "rate": 0.3}
		 ),
	'I': ({"type": "Linear", "out_features": 512},
		  {"type": "Dropout", "rate": 0.4}
		 ),
	'J': ({"type": "Linear", "out_features": 1024},
		  {"type": "Dropout", "rate": 0.5}
		 ),
	'K': ({"type": "Linear", "out_features": 32},
		  {"type": "Dropout", "rate": 0.2}
		 ),
	'L': ({"type": "Linear", "out_features": 32},
		  {"type": "Dropout", "rate": 0.2},
		  {"type": "Linear", "out_features": 32},
		  {"type": "Dropout", "rate": 0.2}
		 ),
	'M': ({"type": "Linear", "out_features": 32},
		  {"type": "Dropout", "rate": 0.2},
		  {"type": "Linear", "out_features": 64},
		  {"type": "Dropout", "rate": 0.2}
		 )
}


def _get_Args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory containing the dataset splited in train, validation and test folders")
	parser.add_argument("-d","--dense", help = "Model with extra dense layers", type=int, nargs='+')
	parser.add_argument("-do","--dropout", help = "Dropout in all dense layers", type=float, nargs='+')
	parser.add_argument("-opt","--optimizer", help = "Optimization algorithm", default='adam', choices=['adam', 'sgd', 'rmsprop'])
	parser.add_argument("-opt_params","--optimizer_parameters", help = "Optimizer parameters", nargs='*')
	parser.add_argument("-e","--epochs", help = "Number of times to run the entire dataset", type=int, default=5)
	parser.add_argument("-lr","--learning_rate", help = "Learning rate", type=float, default=1e-3)
	parser.add_argument("-bs","--batch_size", help = "Number of samples presented by iteration", type=int , default=16)
	parser.add_argument("-kr","--kernel_regularizer", help = "Rate for kernel regularization", type=float)
	parser.add_argument("-l","--log", help = "Log file", default='simple_model_log')
	parser.add_argument("-rm","--resume_model", help = "Name of model to load")
	parser.add_argument("-tboard","--tensorboard", help = "If present save run info to show in tensorboard", action='store_true')
	parser.add_argument("-sp","--show_plots", help = "If present the plots of loss and accuracy will be shown", action='store_true')
	parser.add_argument("-cfg","--config", help = "Use different architecture from hardcoded configurations", nargs='+')
	parser.add_argument("-tl","--transfer_learning", help = "Use with resume model to transfer the weights and train with a low learning rate", action='store_true')
	parser.add_argument("-nc","--nb_class", help = "Number of classes of the previous model", type=int)
	parser.add_argument("-ft","--fine_tunning", help = "Number of layers to unfrozen", type=int)
	
	
	return parser.parse_args()
	
def myPrint(s):
	# funcao para salvar a sumarizacao da rede em um arquivo de texto
	with open('_simple_model_summary.txt','a') as f:
		f.write(s+'\n')
	
def formatTime(t):
	# funcao para transformar a contagem de tempo num formato mais legivel
	if t > 59:
		if t/60 > 59:
			if t/3600 > 23:
				return "{:d}d{:d}h{:d}m{:2.2f}s".format(int(t//86400), int((t%86400)//3600),int(((t%86400)%3600)//60),float(t%60))
			return "{:d}h{:d}m{:2.2f}s".format(int(t//3600),int((t%3600)//60),float(t%60))
		# menos de 60 minutos
		return "{:d}m{:2.2f}s".format(int(t//60),float(t%60))
	# menos de 60 segundos
	return "{:2.2f}s".format(float(t))
	
def get_dirs(dir):

	sets = ['training', 'valid', 'test']
	return os.path.join(dir, sets[0]), os.path.join(dir, sets[1]), os.path.join(dir, sets[2])
	
def handle_opt_params(opt, opt_params):

	optmizers_dict = {
		'adam': optimizers.Adam,
		'sgd': optimizers.SGD,
		'rmsprop': optimizers.RMSprop
	}
	# realiza a conversao de tipos necesaria para os parametros do otimizador de acordo com o otimizador
	if opt_params:
		if opt == 'adam':
			if len(opt_params) > 4:
				opt_params[:4] = map(float, opt_params[:4])
				opt_params[4] = bool(opt[4])
			else:
				opt_params = map(float, opt_params)
		elif opt == 'rmsprop':
				opt_params = map(float, opt_params)
		elif opt == 'sgd':
			if len(opt_params) > 2:
				opt_params[:2] = map(float, opt_params[:2])
				opt_params[2] = bool(opt[2])
			else:
				opt_params = map(float, opt_params)
			
		return optmizers_dict[opt], opt_params
	
	return optmizers_dict[opt], []
	
def _make_layers(X, cfg, batch_norm, kernel_reg, dpout = None):
	
	# cria as camadas do modelo de acordo com o parametro de configuracao passado
	for layer in cfg:
	
		if layer["type"] == "MaxPool2d":
			X = MaxPooling2D(layer["size"], strides = layer["stride"])(X)
			
		elif layer["type"] == "Dropout":
			if dpout:
				X = Dropout(rate = dpout , seed=1)(X)
			else:
				X = Dropout(rate = layer["rate"] , seed=1)(X)
			
		else:
			# If become possible another type of layer, this logic needs to be changed
			if layer["type"] == "Conv2d":
				if kernel_reg:
					X = Conv2D(layer["filters"], kernel_size=layer["size"], strides = layer["stride"], padding = layer["padding"], kernel_regularizer=regularizers.l2(kernel_reg), use_bias=False)(X)
				else:
					X = Conv2D(layer["filters"], kernel_size=layer["size"], strides = layer["stride"], padding = layer["padding"], use_bias=False)(X)
			
			elif layer["type"] == "Linear":
				if kernel_reg:
					X = Dense(layer["out_features"], kernel_regularizer=regularizers.l2(kernel_reg), use_bias=False)(X)
				else:
					X = Dense(layer["out_features"], use_bias=False)(X)
			
			if batch_norm:
				if len(X.shape) > 2:
					X = BatchNormalization(axis = 3)(X)
				else:
					X = BatchNormalization()(X)
					
			X = Activation('relu')(X)		
			
	return X
	
def model_from_config(input_shape, num_classes, kernel_reg, cfg0, cfg1, dpout = None, batch_norm=True, dense = None):

	# monta do modelo de acordo com a opcao de configuracao escolhida para cada parte do modelo
	X_input = Input(input_shape)
	cfg0 = cfg0.upper()
	cfg1 = cfg1.upper()
	
	X = X_input
		
	X = _make_layers(X, CFG_FEATURES[cfg0], batch_norm, kernel_reg, dpout)
	X = Flatten()(X)
	X = _make_layers(X, CFG_CLASSIFIER[cfg1], batch_norm, kernel_reg, dpout)
	# adiciona mais camadas densas alem das presentes na configuracao passada
	# mais tarde preciso alterar para receber uma lista de dropouts para poder configurar taxas de dropout diferentes para cada camada
	if dense:
		for units in dense:
			if kernel_reg:
				X = Dense(units, kernel_regularizer=regularizers.l2(kernel_reg), use_bias=False)(X)
			else:
				X = Dense(units, use_bias=False)(X)
			
			X = BatchNormalization()(X)
			
			X = Activation('relu')(X)
			
			if dpout:
				X = Dropout(dpout)(X)
				
	X = Dense(num_classes, activation='softmax', use_bias=False)(X)

	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = Model(inputs = X_input, outputs = X, name='Model'+cfg0+cfg1)
	
	return model
	
def model(input_shape, num_classes, dpout_rate):
	# montagem do modelo hardcoded que estava usando no comeco
	X_input = Input(input_shape)
	
	X = X_input
	
	############################# CONV-POOL lAYER 1 #############################
	
	X = Conv2D(32, (1, 1), strides = (1, 1), name = 'block0_conv0', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block0_bn0')(X)
	X = Activation('relu')(X)
	
	X = Conv2D(32, (3, 1), strides = (1, 1), name = 'block0_conv1', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block0_bn1')(X)
	X = Activation('relu')(X)
	
	X = Conv2D(64, (1, 3), strides = (1, 1), name = 'block0_conv2', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block0_bn2')(X)
	X = Activation('relu')(X)
	
	'''X = Conv2D(32, (7, 7), strides = (2, 2), name = 'block0_conv0', padding = 'valid', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block0_bn0')(X)
	X = Activation('relu')(X)'''
	
	# MAXPOOL
	X = MaxPooling2D((3, 3), strides = (2, 2), name='max_pool0')(X)
	X = Dropout(rate = dpout_rate, seed=1)(X)

	############################# CONV-POOL lAYER 2 #############################
    
	X = Conv2D(64, (3, 1), strides = (1, 1), name = 'block1_conv0', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block1_bn0')(X)
	X = Activation('relu')(X)
	
	X = Conv2D(128, (1, 3), strides = (1, 1), name = 'block1_conv1', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block1_bn1')(X)
	X = Activation('relu')(X)

	# MAXPOOL
	X = MaxPooling2D((2, 2), strides = (2, 2), name='max_pool1')(X)
	X = Dropout(rate = dpout_rate, seed=1)(X)
	
    
	############################# CONV-POOL lAYER 3 #############################
    
	X = Conv2D(128, (3, 1), strides = (1, 1), name = 'block2_conv0', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block2_bn0')(X)
	X = Activation('relu')(X)
	
	X = Conv2D(256, (1, 3), strides = (1, 1), name = 'block2_conv1', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block2_bn1')(X)
	X = Activation('relu')(X)

	# MAXPOOL
	X = MaxPooling2D((2, 2), strides = (2, 2), name='max_pool2')(X)
	X = Dropout(rate = dpout_rate, seed=1)(X)
	
	
	############################# CONV-POOL lAYER 4 #############################
    
	X = Conv2D(256, (3, 1), strides = (1, 1), name = 'block3_conv0', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block3_bn0')(X)
	X = Activation('relu')(X)
	
	X = Conv2D(512, (1, 3), strides = (1, 1), name = 'block3_conv1', padding = 'same', use_bias=False)(X)
	X = BatchNormalization(axis = 3, name = 'block3_bn1')(X)
	X = Activation('relu')(X)

	# MAXPOOL
	X = MaxPooling2D((2, 2), strides = (2, 2), name='max_pool3')(X)
	X = Dropout(rate = dpout_rate, seed=1)(X)
	
	############################# FC lAYER 1 #############################
    
	# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
	X = Flatten()(X)
	
	X = Dense(1024, name='fc0', use_bias=False)(X)
	X = BatchNormalization(name = 'bn4')(X)
	X = Activation('relu')(X)
	X = Dropout(rate = 0.8, seed=1)(X)
    
	############################# FC lAYER 2 #############################
	
	X = Dense(2048, name='fc1', use_bias=False)(X)
	X = BatchNormalization(name = 'bn5')(X)
	X = Activation('relu')(X)
	X = Dropout(rate = 0.8, seed=1)(X)
	
	############################# FC lAYER 3 #############################
	
	'''X = Dense(4096, name='fc2', use_bias=False)(X)
	X = BatchNormalization(name = 'bn6')(X)
	X = Activation('relu')(X)
	X = Dropout(rate = 0.9, seed=1)(X)'''
	
	############################# FC lAYER 4 #############################
    
	X = Dense(num_classes, activation='softmax', name='fc3', use_bias=False)(X)

	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = Model(inputs = X_input, outputs = X, name='SimpleModel')

    
	return model
	
def schedule(epoch, lr):
	# diminui o alfa em 10 vezes quando chega na epoca 200 
    if epoch == 200:
        lr = lr/10.0
    return lr


def run_CNN(args):

	dir_train, dir_valid, dir_test = get_dirs(args.dir)
	
	global num_classes, num_train, num_test, num_valid
	
	scale_ratios = [1.0, 0.875, 0.75, 0.66]
	
	num_classes = len(os.listdir(dir_train))
	
	num_train = sum([len(files) for r, d, files in os.walk(dir_train)]) 
	num_test = sum([len(files) for r, d, files in os.walk(dir_test)])
	num_valid = sum([len(files) for r, d, files in os.walk(dir_valid)])
	
	#train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
	#train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.3, zoom_range=0.3, rotation_range=0.3)	
	train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
	#train_datagen = ImageDataGenerator(rescale=1./255)
	
	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	#train_datagen.fit(x_train)
	
	train_gen = train_datagen.flow_from_directory(dir_train, target_size = (img_size, img_size), batch_size = args.batch_size, class_mode = 'categorical')
	
	train_gen = train_generator(train_gen, img_size, scale_ratios,num_channels)
	#train_gen = valid_generator(train_gen, img_size, num_channels, save_to_dir= "imgs\\train")
	
	#print("train generator len: "+str(len(train_generator)))
	
	#train_generator = crop_generator(train_generator, img_size, channels = num_channels)
	
	#print("crop generator len: "+str(len(list(train_generator))))

	try:
		test_datagen = ImageDataGenerator(rescale=1./255)
		test_gen = test_datagen.flow_from_directory(dir_test, target_size = (img_size, img_size), batch_size = args.batch_size, class_mode = 'categorical')
		test_gen = valid_generator(test_gen, img_size,num_channels)
	except:
		test_gen = None
	
	valid_datagen = ImageDataGenerator(rescale=1./255)
	valid_gen = valid_datagen.flow_from_directory(dir_valid, target_size = (img_size, img_size), batch_size = args.batch_size, class_mode = 'categorical') 
	#valid_gen = valid_generator(valid_gen, img_size, num_channels, save_to_dir= "imgs\\valid")
	valid_gen = train_generator(valid_gen, img_size, scale_ratios, num_channels)
	
	print('Dataset loaded.')
	
	count = 0
	'''
	names = train_generator.filenames
	while True:
		tempx, tempy = next(crop_gen)
		
		side = int(sqrt(args.batch_size))
		
		print("Shape: {}".format(str(tempx.shape)))
		fig,axes = plt.subplots(nrows = side, ncols = side, figsize=(129,129))
		
		print(names[count*args.batch_size : (count+1)*args.batch_size])
		
		item = train_generator[count]
		#print("type of item: {}".format(type(item)))
		#test = np.array(item)
		#print("Len item: {}".format(len(item)))
		#print(item[1])
		for i in range(4):
			for j in range(4):
				axes[i,j].imshow(item[0][i*4 + j])
			
		plt.show()
		
		count+=1
		for i in range(side):
			for j in range(side):
				axes[i,j].imshow(tempx[i*side + j])
		
		
		plt.show()
		#input()
	'''
	#Create architecture
	if args.transfer_learning:
		nb_classes = args.nb_class
	else:
		nb_classes = num_classes
	
	if args.config:
		if args.dropout:
			simple_model = model_from_config((img_size, img_size, num_channels), nb_classes, args.kernel_regularizer, *args.config, *args.dropout, batch_norm=True)
		else:
			simple_model = model_from_config((img_size, img_size, num_channels), nb_classes, args.kernel_regularizer, *args.config, batch_norm=True)
	else:
		simple_model = model((img_size, img_size, num_channels), nb_classes, *args.dropout)
		
	'''if args.kernel_regularizer:		
		for layer in simple_model.layers:
			if hasattr(layer, 'kernel_regularizer'):
				layer.kernel_regularizer= regularizers.l2(args.kernel_regularizer)
	'''
	
	simple_model.summary(print_fn = myPrint)
	
	#plot_model(simple_model, to_file='simple_model.png')
	
	print('Simple model loaded.')
	
	# carregar modelo previamente treinado com os mesmos parâmetros passados
	if args.resume_model:
		simple_model.load_weights(args.resume_model)
		print('Weights loaded')

	# transferencia de pesos quando o tamanho do softmax muda
	if args.transfer_learning:
		tf_model = model_from_config((img_size, img_size, num_channels), num_classes, args.kernel_regularizer, *args.config, *args.dropout, batch_norm=True, dense = args.dense)
		extra_layers = 0
		if args.dense:
			extra_layers = len(args.dense)
		simple_model = transferWeights(simple_model, tf_model, 0, len(simple_model.layers)-1)
		'''for layer in simple_model.layers[:-1-extra_layers]:
			layer.trainable = False'''
		
		for layer in simple_model.layers:
			print("Layer: {} Trainable: {}".format(layer, layer.trainable))
			
	if args.fine_tunning:
		uf_layers = 2 + 4*args.fine_tunning
		for layer in simple_model.layers[:-uf_layers]:
			layer.trainable = False
		
		for layer in simple_model.layers:
			print("Layer: {} Trainable: {}".format(layer, layer.trainable))
	
	opt, opt_params = handle_opt_params(args.optimizer, args.optimizer_parameters)
	simple_model.compile(opt(args.learning_rate, *opt_params), loss = "categorical_crossentropy" , metrics=['accuracy'])	
		
	print('Model compiled.')

	csv_logger = CSVLogger(args.log+'.csv', append=True, separator=';')
	#Save the model after every epoch.
	checkpoint_path = 'simple_model_best_resize_lrate'+str(args.learning_rate)+'_bsize'+str(args.batch_size)+'_epochs'+str(args.epochs)+'_opt_'+args.optimizer+'_'+str(time.time())+'.h5'
	mc_best = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	lrs = LearningRateScheduler(schedule, verbose=0)
	
	if args.tensorboard:
		tb = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=args.batch_size, write_graph=True, write_grads=False, write_images=True)
		hist = simple_model.fit_generator(train_gen, steps_per_epoch = num_train // args.batch_size, epochs = args.epochs, callbacks=[csv_logger, tb, mc_best], validation_data=valid_gen, validation_steps=num_valid // args.batch_size, shuffle = True)
	else:
		hist = simple_model.fit_generator(train_gen, steps_per_epoch = num_train // args.batch_size, epochs = args.epochs, callbacks=[csv_logger, mc_best], validation_data=valid_gen, validation_steps=num_valid // args.batch_size, shuffle = True)
	
	
	if test_gen:
		score = simple_model.evaluate_generator(test_gen, steps = num_test // args.batch_size)
	else:
		score = "No set to test"
		
	#model_name = 'model_batchnorm_kernelreg'+str(kernel_reg)+'_media_lrate'+str(learning_rate)+'_bsize'+str(batch_size)+'_dpout'+str(dpout)+'_epochs'+str(epochs)+'_opt_'+opt+'.h5'
	model_name = 'simple_model_resize_lrate'+str(args.learning_rate)+'_bsize'+str(args.batch_size)+'_epochs'+str(args.epochs)+'_opt_'+args.optimizer+'_'+str(time.time())+'.h5'
	
	simple_model.save_weights(model_name)
	
	return score, hist, model_name, checkpoint_path
	
def plot_and_save(history, name, show_plots):

	os.makedirs("imgs", exist_ok=True)

	# summarize history for accuracy
	fig, ax = plt.subplots()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'])
	
	if show_plots:
		plt.show()
	
	# summarize history for loss
	fig2, ax2 = plt.subplots()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'])
	
	if show_plots:
		plt.show()
	
	
	fig.savefig("imgs/"+name[:-3]+"_acc.jpg")
	fig2.savefig("imgs/"+name[:-3]+"_loss.jpg")
	
	return name[:-3]+"_acc.jpg", name[:-3]+"_loss.jpg"
	
def print_best_acc(history):
	
	idx = np.argmax(history.history['val_acc'])
	
	print("Best accuracy (epoch {}): \nloss: {:6.4f} acc: {:6.4f} val_loss: {:6.4f} val_acc: {:6.4f} ".format(idx+1, history.history['loss'][idx], history.history['acc'][idx], history.history['val_loss'][idx], history.history['val_acc'][idx]))
	
	return idx
	
def _main(args):
	# TODO: PASSAR TUDO QUE FOR NECESSARIO PARA A FUNCAO SAVE_INFOS COMO RETORNO ATE CHEGAR NAQUELE IF ALI EMBAIXO
	score, hist, name_weights, name_weights_best = run_CNN(args)
	print(score)
	names = plot_and_save(hist, name_weights, args.show_plots)
	idx = print_best_acc(hist)
	
	return score, hist, names, idx, name_weights, name_weights_best
 

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	
	start = time.ctime()
	print("\n*********\nBegin time: "+start+"\n*********\n")
	s = time.time()
	score, hist, names, idx, nm_w, nm_w_b = _main(args)
	t = time.time()
	end = time.ctime()
	print("\n*********\nEnd time: "+end+"\n*********\n")
	time_formated = formatTime(t-s)
	print("\n########\nElapsed time: " + time_formated+"\n########\n")
	save_infos(os.path.basename(__file__), args, hist, idx, score, nm_w, nm_w_b, *names, [start, end, time_formated], "G:\Meu Drive\Mestrado\Experimentos")
	

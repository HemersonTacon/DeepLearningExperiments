#Keras import
from keras import applications, optimizers, regularizers
from keras.callbacks import CSVLogger
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from fold import read_folder_structure, create_folder_structure


import os
import numpy as np
import argparse
import time
import sys
import time


#Params
c_valid = 'custom' #kfold, ekfold, traintest
k = 10 #K in kfold
percent = 0.25  #percent in train test split
percent_valid = 0.15
batch_size = 16
img_size = 299
num_classes = 0
num_channels = 3
learning_rate = 1e-3
epochs = 10
dpout = 0.7
opt = 'adam'
kernel_reg = 0.01
num_train = 0 #Number of samples in train, valid and test 
num_valid = 0
num_test = 0
validation = True


def _get_Args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory containing the dataset splited in train, validation and test folders")
	parser.add_argument("-d","--dense", help = "Model with extra dense layers", type=int, nargs='+')
	parser.add_argument("-do","--dropout", help = "Dropout in all dense layers", type=float	, default=0.0, nargs='+')
	parser.add_argument("-opt","--optimizer", help = "Optimization algorithm", default='adam', choices=['adam', 'sgd', 'rmsprop'])
	parser.add_argument("-opt_params","--optimizer_parameters", help = "Optimizer parameters", nargs='*')
	parser.add_argument("-e","--epochs", help = "Number of times to run the entire dataset", type=int, default=5)
	parser.add_argument("-lr","--learning_rate", help = "Learning rate", type=float, default=1e-3)
	parser.add_argument("-bs","--batch_size", help = "Number of samples presented by iteration", type=int , default=16)
	parser.add_argument("-kr","--kernel_regularizer", help = "Rate for kernel regularization", type=float, default=0.0)
	parser.add_argument("-l","--log", help = "Log file", default='log10')
	parser.add_argument("-rm","--resume_model", help = "Name of model to load")
	
	return parser.parse_args()
	

	
def formatTime(t):
	
	if t > 59:
		if t/60 > 59:
			if t/3600 > 23:
				return "{:d}d{:d}h{:d}m{:2.2f}s".format(int(t//86400), int((t%86400)//3600),int(((t%86400)%3600)//60),float(t%60))
			return "{:d}h{:d}m{:2.2f}s".format(int(t//3600),int((t%3600)//60),float(t%60))
		# menos de 60 minutos
		return "{:d}m{:2.2f}s".format(int(t//60),float(t%60))
	# menos de 60 segundos
	return "{:2.2f}s".format(float(t))

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
	
def transferWeights(old_m, new_m, begin_l, end_l):

	for i in range(begin_l, end_l):
		w = old_m.layers[i].get_weights()
		new_m.layers[i].set_weights(w)
		
	return new_m
	
def get_dirs(dir):

	sets = ['training', 'valid', 'test']
	
	return os.path.join(dir, sets[0]), os.path.join(dir, sets[1]), os.path.join(dir, sets[2])
	
def handle_opt_params(opt, opt_params):

	optmizers_dict = {
		'adam': optimizers.Adam,
		'sgd': optimizers.SGD,
		'rmsprop': optimizers.RMSprop
	}
	
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


def run_CNN(args):

	dir_train, dir_valid, dir_test = get_dirs(args.dir)
	
	global num_classes, num_train, num_test, num_valid
	
	num_classes = len(os.listdir(dir_train))
	
	num_train = sum([len(files) for r, d, files in os.walk(dir_train)]) 
	num_test = sum([len(files) for r, d, files in os.walk(dir_test)])
	num_valid = sum([len(files) for r, d, files in os.walk(dir_valid)])
	
	#train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
	#train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.3, zoom_range=0.3, rotation_range=0.3)	
	train_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory(dir_train, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical')

	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(dir_test, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical')
	
	valid_datagen = ImageDataGenerator(rescale=1./255)
	valid_generator = valid_datagen.flow_from_directory(dir_valid, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical') 
	
	print('Dataset loaded.')

	base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape = (img_size, img_size, num_channels))	
	
	print('Base model loaded.')


	#Create architecture

	#x = base_model.output
	#x = Flatten()(x)
	#x = Dense(64, activation='relu')(x) 
	#predictions = Dense(num_classes, activation="softmax")(x)
	
	x = base_model.output
	# Rectangular convolution with same padding and maintaining the same number of channels
	#x = Conv2D(2048, (1, 3), padding='same', activation='relu', use_bias=False)(x)
	#x = GlobalAveragePooling2D()(x)
	#x = Dense(64, activation='relu')(x)
	
	if args.kernel_regularizer > 0:
			x = Conv2D(1024, (1,1), padding='valid', kernel_regularizer=regularizers.l2(args.kernel_regularizer), use_bias=False)(x)
	else:
		x = Conv2D(1024, (1,1), padding='valid', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	
	x = GlobalAveragePooling2D()(x)
	
	#x = Flatten()(x)
	
	if not args.dropout == 0.0 and len(args.dropout) == 1:
		args.dropout = [args.dropout[0]]*len(args.dense)
	elif type(args.dropout) == float:
		args.dropout = [args.dropout]*len(args.dense)
	
	for units, rate in zip(args.dense, args.dropout):
		if args.kernel_regularizer > 0:
			x = Dense(units, kernel_regularizer=regularizers.l2(args.kernel_regularizer))(x)
		else:
			x = Dense(units)(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		if rate > 0.0:
			x = Dropout(rate)(x)
		
	if args.kernel_regularizer > 0:
		x = Dense(num_classes, kernel_regularizer=regularizers.l2(args.kernel_regularizer))(x)
	else:
		x = Dense(num_classes)(x)
	
	# Batch normalization antes do classificador não surtiu muito efeito positivo
	#x = BatchNormalization()(x)
	predictions = Activation('softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	
	for layer in base_model.layers:
		layer.trainable = False
		
	# seção para descongelar os resto da rede para depois fazer finetunning de fato
	'''
	for layer in model.layers[172:]:
		layer.trainable = True 
	'''
	
	# carregar modelo previamente treinado com os mesmos parâmetros passados
	if args.resume_model:
		model.load_weights(args.resume_model)
		print('from disk')
	
	# transferencia de pesos quando o tamanho do softmax muda
	#model = transferWeights(old_model, new_model, len(base_model.layers), len(old_model.layers)-1)
	
	opt, opt_params = handle_opt_params(args.optimizer, args.optimizer_parameters)
	model.compile(opt(args.learning_rate, *opt_params), loss = "categorical_crossentropy" , metrics=['accuracy', top_3_accuracy])	
		
	#model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss = "categorical_crossentropy" , metrics=['accuracy', top_3_accuracy])	
	print('Model loaded.')

	csv_logger = CSVLogger(args.log+'.csv', append=True, separator=';')

	model.fit_generator(train_generator, steps_per_epoch = num_train // args.batch_size, epochs = args.epochs, callbacks=[csv_logger], validation_data=valid_generator, validation_steps=num_valid // args.batch_size, shuffle = True)

	
	#model_name = 'model_batchnorm_kernelreg'+str(kernel_reg)+'_media_lrate'+str(learning_rate)+'_bsize'+str(batch_size)+'_dpout'+str(dpout)+'_epochs'+str(epochs)+'_opt_'+opt+'.h5'
	model_name = 'model_dense'+str(args.dense)+'_batchnorm_media_lrate'+str(args.learning_rate)+'_bsize'+str(args.batch_size)+'_dpout'+str(args.dropout)+'_epochs'+str(args.epochs)+'_opt_'+args.optimizer+'.h5'
	
	model.save_weights(model_name)
	score = model.evaluate_generator(test_generator, steps = num_test // args.batch_size)
	
	return score


def _main(args):
		
	score = run_CNN(args)
	print(score)
 

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	
	print("\n*********\nBegin time: "+time.ctime()+"\n*********\n")
	s = time.time()
	_main(args)
	t = time.time()
	print("\n*********\nEnd time: "+time.ctime()+"\n*********\n")
	print("\n########\nElapsed time: " + formatTime(t-s)+"\n########\n")

#Keras import
from keras import applications, optimizers
from keras.callbacks import CSVLogger
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from fold import read_folder_structure, create_folder_structure


import os
import numpy as np
import argparse
import time
import sys
import pprint
import itertools
from math import ceil
from simpleModel import model_from_config

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
		 )
}

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
		 )
}


#Params
c_valid = 'custom' #kfold, ekfold, traintest
k = 10 #K in kfold
percent = 0.25  #percent in train test split
percent_valid = 0.15
batch_size = 8
img_size = 129
num_classes = 0
num_channels = 3
learning_rate = 1e-3
epochs = 10
num_train = 0 #Number of samples in train, valid and test 
num_valid = 0
num_test = 0
validation = True


def _get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory containing the dataset splited in train, validation and test folders")
	parser.add_argument("model", help="Model weights to predict samples")
	parser.add_argument("--load", "-l", action= 'store_true', help="Load already predicted labels if present")
	parser.add_argument("-cfg","--config", help = "Use different architecture from hardcoded configurations", nargs='+')
	parser.add_argument("-do","--dropout", help = "Dropout in all dense layers", type=float, nargs='+')
	return parser.parse_args()
	
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
def get_dirs(dir):

	sets = ['training', 'valid', 'test']
	
	return os.path.join(dir, sets[0]), os.path.join(dir, sets[1]), os.path.join(dir, sets[2])
	
def prepare_model(model_name):

	# carregar modelo previamente treinado com os mesmos parâmetros passados
	if args.resume_model:
		simple_model.load_weights(args.resume_model)
		print('Weights loaded')
		
	
	base_model = applications.InceptionV3(weights=None, include_top=False, input_shape = (img_size, img_size, num_channels))	
	
	print('Base model loaded.')
	
	x = base_model.output
	
	x = Conv2D(2048, (1, 3), padding='same', activation='relu', use_bias=False)(x)
	x = GlobalAveragePooling2D()(x)
	
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.8)(x)
	predictions = Dense(16, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)
	
	print("Loading weights")
	model.load_weights(model_name)
	
	print('Weights loaded.')
	
	model.compile(optimizer='rmsprop', loss = "categorical_crossentropy" , metrics=['accuracy'])
	
	return model
	
def prepare_model_from_config(config, dropout, num_classes, model_name, kernel_reg = None):

	print("Configs:")
	print(config)

	if dropout:
			simple_model = model_from_config((img_size, img_size, num_channels), num_classes, kernel_reg, *config, *dropout, batch_norm=True, None)
	else:
		simple_model = model_from_config((img_size, img_size, num_channels), num_classes, kernel_reg, *config, batch_norm=True, None)
			
	simple_model.load_weights(model_name)
	print('Weights loaded')
	
	return simple_model
	
def get_labels(dir_train, dir_valid, dir_test, model, load):
	
	num_train = sum([len(files) for r, d, files in os.walk(dir_train)]) 
	train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
	train_generator = train_datagen.flow_from_directory(dir_train, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical')
	train_labels = train_generator.classes
	
	try:
		if not load:
			raise Exception
		train_pred_labels = np.load('train_pred_labels.npy')
	except:
		predictions = model.predict_generator(train_generator, steps = ceil(num_train/batch_size))
		train_pred_labels = predictions.argmax(axis=-1)
		np.save('train_pred_labels.npy', train_pred_labels)
	
	cm = confusion_matrix(train_labels, train_pred_labels)
	plt.figure()
	plot_confusion_matrix(cm, train_generator.class_indices.keys(), title='Confusion matrix - Training')
	
	arr = list(map(float, np.equal(train_labels, train_pred_labels)))
	train_acc = sum(arr)/len(arr)
	
	print("Train accuracy: {:5.2f}%".format(train_acc*100))
	
	plt.show()
	
	
	
	if os.path.isdir(dir_valid):
		# counting samples
		num_valid = sum([len(files) for r, d, files in os.walk(dir_valid)])
		# normalizing images
		valid_datagen = ImageDataGenerator(rescale=1./255)
		valid_generator = valid_datagen.flow_from_directory(dir_valid, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical') 
		# getting the ground truth labels
		valid_labels = valid_generator.classes	
		
		try:
			if not load:
				raise Exception
			valid_pred_labels = np.load('valid_pred_labels.npy')
		except:
			predictions = model.predict_generator(valid_generator, steps = ceil(num_valid/batch_size))
			# transforming probabilties into labels
			valid_pred_labels = predictions.argmax(axis=-1)
			np.save('valid_pred_labels.npy', valid_pred_labels)
		
		cm = confusion_matrix(valid_labels, valid_pred_labels)
		plt.figure()
		plot_confusion_matrix(cm, valid_generator.class_indices.keys(), title='Confusion matrix - Validation')
		
		arr = list(map(float, np.equal(valid_labels, valid_pred_labels)))
		valid_acc = sum(arr)/len(arr)
		
		print("Train accuracy: {:5.2f}%".format(valid_acc*100))
		
		plt.show()
		
		
	
	#print('Dataset loaded.')
	
	if os.path.isdir(dir_test):
		num_test = sum([len(files) for r, d, files in os.walk(dir_test)])
		test_datagen = ImageDataGenerator(rescale=1./255)
		test_generator = test_datagen.flow_from_directory(dir_test, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical')
		test_labels = test_generator.classes
		
		try:
			if not load:
				raise Exception
			test_pred_labels = np.load('test_pred_labels.npy')
		except:
			predictions = model.predict_generator(test_generator, steps = ceil(num_test/batch_size))
			test_pred_labels = predictions.argmax(axis=-1)
			np.save('test_pred_labels.npy', test_pred_labels)
			
		cm = confusion_matrix(test_labels, test_pred_labels)
		plt.figure()
		plot_confusion_matrix(cm, test_generator.class_indices.keys(), title='Confusion matrix - Testing')
		
		arr = list(map(float, np.equal(test_labels, test_pred_labels)))
		test_acc = sum(arr)/len(arr)
		
		print("Train accuracy: {:5.2f}%".format(test_acc*100))
		
		plt.show()
		
		
	
def _main(args):

	dir_train, dir_valid, dir_test = get_dirs(args.dir)
	
	num_classes = len(os.listdir(dir_train))
	
	if args.config:
		model = prepare_model_from_config(args.config, args.dropout, num_classes, args.model)
	else:
		model = prepare_model(args.model)

	get_labels(dir_train, dir_valid, dir_test, model, args.load)
	
	
if __name__ == "__main__":

	_main(_get_args())
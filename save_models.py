#Keras import
from keras.models import model_from_json
#from keras import applications, optimizers, regularizers
#from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Input
from keras.models import Model, Sequential
#from keras.preprocessing.image import ImageDataGenerator
#from keras.metrics import top_k_categorical_accuracy
#from keras.utils.vis_utils import plot_model
#import matplotlib.pyplot as plt
#from transform import valid_generator, train_generator
#from finetunninginception import transferWeights
#from automatize_helper import save_infos

import os
import numpy as np
import argparse
#import time
#import sys
#import time
#import cv2
#from math import sqrt
from simpleModel import model_from_config
#from pprint import pprint
import json
from tqdm import tqdm 
from keras import backend as K


# tamanho antes do crop
#img_size_for_crop = 192
# tamanho da entrada da rede

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
		 ),
	'I': ({"type": "Conv2d", "filters": 8, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 8, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 16, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 16, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 32, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 32, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 64, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
		  {"type": "Conv2d", "filters": 64, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 128, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2}
		 ),
	'J': ({"type": "Conv2d", "filters": 8, "size":(1,1), "stride": 1, "padding": "valid"},
		  {"type": "Conv2d", "filters": 8, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 8, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 3, "stride": 2},
		  {"type": "Conv2d", "filters": 16, "size":(3,1), "stride": 1, "padding": "same"},
		  {"type": "Conv2d", "filters": 16, "size":(1,3), "stride": 1, "padding": "same"},
		  {"type": "MaxPool2d", "size": 2, "stride": 2},
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
		  {"type": "Dropout", "rate": 0.8}
		 ),
	'G': ({"type": "Linear", "out_features": 128},
		  {"type": "Dropout", "rate": 0.8}
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
		  {"type": "Dropout", "rate": 0.8}
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
	parser.add_argument("dir", help="Directory where the model configurations will be saved")
	parser.add_argument("-ext", "--extension", help = "Model extension", default='json', choices=['json','yaml'])	
	
	return parser.parse_args()
	
def _main(dir, ext):
	
	img_sizes = [129, 193, 227, 299]
	num_channels = [3,5]
	num_classes = [17, 101]
	total_configs = len(img_sizes)*len(num_channels)*len(CFG_FEATURES)*len(CFG_CLASSIFIER)
	#process = psutil.Process(os.getpid())
	
	with tqdm(total=total_configs, ascii=True) as pbar:
		for img_size in img_sizes:
			for num_channel in num_channels:
				for num_class in num_classes:
					curr_dir = os.path.join(dir, 'img_size_{}'.format(img_size), 'num_channels_{}'.format(num_channel), 'num_classes_{}'.format(num_class))
					files = os.listdir(curr_dir) if os.path.isdir(curr_dir) else []
					for cfg0 in CFG_FEATURES:
						for cfg1 in CFG_CLASSIFIER:
							pbar.update(1)
							name = 'model_{}_{}.json'.format(cfg0, cfg1)
							# skip already existing files
							if name in files:
								continue
							ml = None
							json_string = None			
							#print("Memory use before load: {}".format(process.memory_info().rss))
							ml = model_from_config((img_size, img_size, num_channel), num_class, None, cfg0, cfg1)
							#print("Memory use after load: {}".format(process.memory_info().rss))
							json_string = ml.to_json()
							
							os.makedirs(curr_dir, exist_ok=True)
							with open(os.path.join(curr_dir, name), "w") as f:
								json.dump(json_string, f)
							K.clear_session()
						

	'''##pprint(cfg)
	json_string = ml.to_json()
	#pprint(json_string)
	with open("out.json", "w") as f:
		json.dump(json_string, f)
	
	with open("out.json", "r") as f:
		json_string = json.loads(f.read())
		
	ml = model_from_json(json_string)
	cfg = ml.get_config()
	pprint(cfg)'''

if __name__ == '__main__':

	args = _get_Args()	
	_main(args.dir, args.extension)
	
	
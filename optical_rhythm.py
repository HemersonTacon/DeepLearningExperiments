import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import re

DIRECTION = {
				'H': 0,
				'V': 1
			}

def _get_Args():

	parser = argparse.ArgumentParser()
	parser.add_argument("input", help = "Directory with a sample flow")
	parser.add_argument("output", help = "Output dir")
	parser.add_argument("-e", "--ext", help= "Extension of output", default=".png")
	#parser.add_argument("-m", "--mode", help= "Capture mode", default="mean", choices=['mean', 'gaussian'])
	#parser.add_argument("-sg", "--sigma", type=float, help= "Standard deviation to use whitin gaussian filter", default=1.0)
	#parser.add_argument("-s", "--size", type=float, help= "Filter size", default=3.0)
	parser.add_argument("-d", "--direction", help= "Visual rhythm direction", choices=["V", "H", "v", "h"], default='H')
	#parser.add_argument("-p", "--percentil", help= "Relative position to extract the rhythm", type=float, default=[0.5], nargs='+')
	#parser.add_argument("-c", "--color_mode", help= "Relative position to extract the rhythm", choices=['rgb', 'gray'], default='gray')

	return parser.parse_args()

def frame_tensor(flow_x, flow_y, direction, flow_shape):

	x2 = flow_x**2
	y2 = flow_y**2
	xy = flow_x*flow_y

	try:
		ax = DIRECTION[direction]
		x2_mean = np.sum(x2,axis=ax)/flow_shape[ax]
		y2_mean = np.sum(y2,axis=ax)/flow_shape[ax]
		xy_mean = np.sum(xy,axis=ax)/flow_shape[ax]
		return np.array([[a2, b2, ab] for a2, b2, ab in zip(x2_mean, y2_mean, xy_mean)])
	except Exception as e:
		print("Check passed direction: {}".format(e))
		exit(0)
	return None

def video_tensor(args):

	args.direction = args.direction.upper()

	# expect folder to have flow x, y and rgb image
	total_frames = len(os.listdir(args.input))//3

	for i in range(total_frames):

		img_x = os.path.join(args.input, "flow_x_{:05d}.jpg".format(i+1))
		flow_x = cv2.imread(img_x)
		if flow_x is None:
			print("Failed to read image {}".format(img_x))
			exit(0)

		img_y = os.path.join(args.input, "flow_y_{:05d}.jpg".format(i+1))
		flow_y = cv2.imread(img_y)
		if flow_y is None:
			print("Failed to read image {}".format(img_y))
			exit(0)

		flow_x = cv2.cvtColor(flow_x, cv2.COLOR_BGR2GRAY)
		flow_y = cv2.cvtColor(flow_y, cv2.COLOR_BGR2GRAY)

		if i == 0:
			height, width = flow_x.shape
			if args.direction == 'H':
				opt_rtm = np.zeros((width, total_frames, 3))
			elif args.direction == 'V':
				opt_rtm = np.zeros((height, total_frames, 3))

		opt_rtm[:,i,:] = frame_tensor(flow_x, flow_y, args.direction, (height, width))

	# create class pattern to extract class name and create it folder
	class_pat = re.compile(r"v_([A-Za-z]+)_g")
	try:
		class_name = class_pat.search(args.input).group(1)
	except AttributeError:
		print("Failed to extract class pattern. Aborting...")
		exit(0)
	output = os.path.join(args.output, class_name)
	os.makedirs(output, exist_ok = True)

	img_name = os.path.split(args.input)[-1] + args.ext

	if args.ext == '.npy':
		np.save(os.path.join(output, img_name), opt_rtm)
	else:
		cv2.imwrite(os.path.join(output, img_name), opt_rtm)

def _main(args):
	video_tensor(args)

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)

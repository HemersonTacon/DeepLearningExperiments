import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

DIRECTION = {
				'H': 0,
				'V': 1
			}

def _get_Args():

	parser = argparse.ArgumentParser()
	parser.add_argument("input", help = "Video file")
	parser.add_argument("output", help = "Output dir")
	parser.add_argument("-e", "--ext", help= "Extension of output", default="jpg")
	parser.add_argument("-m", "--mode", help= "Capture mode", default="mean", choices=['mean', 'gaussian'])
	parser.add_argument("-sg", "--sigma", type=float, help= "Standard deviation to use whitin gaussian filter", default=1.0)
	parser.add_argument("-s", "--size", type=float, help= "Filter size", default=3.0)
	parser.add_argument("-d", "--direction", help= "Visual rhythm direction", choices=["V", "H", "v", "h"], default='H')
	parser.add_argument("-p", "--percentil", help= "Relative position to extract the rhythm", type=float, default=[0.5], nargs='+')
	parser.add_argument("-c", "--color_mode", help= "Relative position to extract the rhythm", choices=['rgb', 'gray'], default='gray')

	return parser.parse_args()

def mean_rhythm(frame, direction=['H']):

	return np.mean(frame, axis=DIRECTION[direction]) # flat video in vertical/horizontal direction

def gaussian_rhythm(frame, sigma, filter_size, width, height, direction='H', percentil=[0.5], color_mode='gray'):

	trunc = (((filter_size - 1)/2)-0.5)/sigma

	first = True
	output = []

	out = gaussian_filter1d(frame, sigma=sigma, axis=DIRECTION[direction], truncate=trunc)

	#print("Output:\n{}".format(out))
	#print("Output:\n{}".format(out.shape))
	for percent in percentil:
		if color_mode == 'gray':
			if direction == 'H':
				vr = out[int(out.shape[0]*percent),:]
			elif direction == 'V':
				vr = out[:,int(out.shape[1]*percent)]
			vr = np.expand_dims(vr, axis=-1)
		elif color_mode == 'rgb':
			if direction == 'H':
				vr = out[int(out.shape[0]*percent),:,:]
			elif direction == 'V':
				vr = out[:,int(out.shape[1]*percent),:]

		if first:
			first = False
			output = vr
			continue
		output = np.concatenate((output, vr), axis=-1)
	'''fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize=(4*2,3*2))
	axes[0].imshow(frame, cmap='gray')
	axes[1].imshow(out, cmap='gray')
	#axes[2].imshow(vr)
	plt.show()'''

	return output

def videoCapture(args):

	args.direction = args.direction.upper()

	# capture video passing video filename
	vid = cv2.VideoCapture(args.input)

	# verifies if it have initialized the capture
	if(not vid.isOpened()): return None

    # obtain video information
	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = float(vid.get(cv2.CAP_PROP_FPS))
	'''print ("video: {}".format(args.input))
	print ("video resolution: {} x {}".format(width, height))
	print ("number of frames: {}".format(length))
	print ("frame rate: {:.2f}".format(fps))'''

	if args.direction == 'H':
		vr = np.zeros((width,length, len(args.percentil) if args.color_mode == 'gray' else len(args.percentil)*3), dtype=np.uint8)
	elif args.direction == 'V':
		vr = np.zeros((height,length, len(args.percentil) if args.color_mode == 'gray' else len(args.percentil)*3), dtype=np.uint8)


	for i in range(length):
	# read frame from video
		(flag, frame) = vid.read()
		if not flag: # check if frame is valid (readed correctly)
			break

		if args.color_mode == 'gray':
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
		elif args.color_mode == 'rgb':
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB

		if args.mode == 'mean':
			vr[:,i,:] = mean_rhythm(frame, args.direction) # setting visual rhythm for frame i
		if args.mode == 'gaussian':
			vr[:,i,:] = gaussian_rhythm(frame, args.sigma, args.size, width, height, args.direction, args.percentil, args.color_mode) # setting visual rhythm for frame i

	# get directory and and get the last subidr of directory
	temp, _class = os.path.split(os.path.dirname(args.input))
	# get the last subidr of directory again
	set = os.path.split(temp)[-1]
	outdir = os.path.join(args.output, set, _class)
	#print("Class {} Set {} Temp {} Outdir {}".format(_class, set, temp, outdir))
	os.makedirs(outdir, exist_ok=True)
	filename = os.path.splitext(os.path.basename(args.input))[0] + "_sz_{}_sg_{}_p_{}.{}".format(args.size, args.sigma, args.percentil, args.ext)
	filename = os.path.join(outdir, filename)

	if args.ext == "npy":
		np.save(filename, vr)
	else:
		cv2.imwrite(filename, vr)

def _main(args):
	videoCapture(args)

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)

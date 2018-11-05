import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from math import exp

DIRECTION = {
				'H': 0,
				'V': 1
			}

def _get_Args():

	parser = argparse.ArgumentParser()
	parser.add_argument("input", help = "Video file")
	parser.add_argument("output", help = "Output dir")
	parser.add_argument("-e", "--ext", help= "Extension of output", default="jpg")
	parser.add_argument("-m", "--mode", help= "Capture mode", default="mean", choices=['mean', 'gaussian', 'transp'])
	parser.add_argument("-sg", "--sigma", type=float, help= "Standard deviation to use whitin gaussian filter")
	parser.add_argument("-s", "--size", type=float, help= "Filter size")
	parser.add_argument("-d", "--direction", help= "Visual rhythm direction", choices=["V", "H", "v", "h"], default='H')
	parser.add_argument("-p", "--percentil", help= "Relative position to extract the rhythm", type=float, default=[0.5], nargs='+')
	parser.add_argument("-c", "--color_mode", help= "Color mode", choices=['rgb', 'gray', 'ic'], default='gray')

	return parser.parse_args()

def gaussian(size, sigma):
    	v = [exp(-((i-size//2)**2/(sigma)**2)) for i in range(int(size))]
    	return 255*(np.array(v)/max(v))

def alpha_img(frame, sigma, filter_size, width, height, frame_n, out):

	filter_size = int(filter_size)
	new_shape = list(frame.shape)
	new_shape[2] = new_shape[2] + 1 # adding alpha channel
	transp_img = np.zeros(tuple(new_shape))
	alpha = gaussian(filter_size, sigma)
	#print(alpha)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	for i in range(filter_size//2):
		transp_img[height//2 + i,:,:] = np.concatenate((frame[height//2 + i,:,:],
									new_shape[1]*[[alpha[filter_size//2 + i]]]),
									axis=-1)
		transp_img[height//2 - i,:,:] = np.concatenate((frame[height//2 - i,:,:],
									new_shape[1]*[[alpha[filter_size//2 - i]]]),
									axis=-1)

	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#transp_img = cv2.cvtColor(transp_img, cv2.COLOR_BGRA2RGBA)
	cv2.imwrite(os.path.join(out, "transp_img{}.png".format(frame_n)), transp_img)
	cv2.imwrite(os.path.join(out, "frame{}.png".format(frame_n)), frame)

def mean_rhythm(frame, direction=['H']):

	return np.expand_dims(np.mean(frame, axis=DIRECTION[direction]), axis=-1) # flat video in vertical/horizontal direction

def gaussian_rhythm(frame, sigma, filter_size, width, height, direction='H', percentil=[0.5], color_mode='gray'):

	trunc = (((filter_size - 1)/2)-0.5)/sigma

	first = True
	output = []

	out = gaussian_filter1d(frame, sigma=sigma, axis=DIRECTION[direction], truncate=trunc)

	#print("Output:\n{}".format(out))
	#print("Output:\n{}".format(out.shape))
	for percent in percentil:

		if direction == 'H':
			vr = out[int(out.shape[0]*percent),:]
		elif direction == 'V':
			vr = out[:,int(out.shape[1]*percent)]

		if color_mode == 'gray':
			vr = np.expand_dims(vr, axis=-1)

		if first:
			first = False
			output = vr
		else:
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
			print("Invalid frame at position {} of {} on video {}".format(i+1,length,args.input))
			break

		if args.color_mode == 'gray':
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
		else:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB

		if args.mode == 'mean':
			vr[:,i,:] = mean_rhythm(frame, args.direction) # setting visual rhythm for frame i
		if args.mode == 'gaussian':
			vr[:,i,:] = gaussian_rhythm(frame, args.sigma, args.size, width, height, args.direction, args.percentil, args.color_mode) # setting visual rhythm for frame i
		if args.mode == "transp":
			alpha_img(frame, args.sigma, args.size, width, height, i, args.output)

	# slices only valid frames
	vr = vr[:,:i]

	# get directory and and get the last subidr of directory
	temp, _class = os.path.split(os.path.dirname(args.input))
	# get the last subidr of directory again
	_set = os.path.split(temp)[-1]
	outdir = os.path.join(args.output, _set, _class)
	#print("Class {} Set {} Temp {} Outdir {}".format(_class, set, temp, outdir))
	os.makedirs(outdir, exist_ok=True)
	if args.sigma or args.percentil:
		filename = os.path.splitext(os.path.basename(args.input))[0] + "_sz_{}_sg_{}_p_{}.{}".format(args.size, args.sigma, args.percentil, args.ext)
	elif args.mode == 'mean':
		filename = os.path.splitext(os.path.basename(args.input))[0] + ".{}".format(args.ext)
	else:
		filename = os.path.splitext(os.path.basename(args.input))[0] + "_p_{}.{}".format(args.percentil, args.ext)


	if args.color_mode == "ic":
		temp = 'RGB'
		for i in range(3):
			temp_out = os.path.join(outdir, temp[i])
			os.makedirs(temp_out, exist_ok=True)
			temp_name = os.path.join(temp_out, filename)
			if args.ext == "npy":
				np.save(temp_name, vr[:,:,i])
			else:
				cv2.imwrite(temp_name, vr[:,:,i])
	else:
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

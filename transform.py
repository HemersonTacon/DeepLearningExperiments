import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image as pil_image


##Data augmentation

def random_crop(img, crop_size):

	assert img.shape[2] == 1 or img.shape[2] == 3

	height, width = img.shape[0], img.shape[1]
	dy, dx = crop_size, crop_size
	x = np.random.randint(0, width - dx + 1)
	y = np.random.randint(0, height - dy + 1)
	  
	return img[y:(y+dy), x:(x+dx), :]


def center_crop(img, crop_size):

	assert img.shape[2] == 1 or img.shape[2] == 3
	
	height, width = img.shape[0], img.shape[1]
	dx,dy = crop_size, crop_size
	x = int((width - dx) / 2.)
	y = int((height - dy) / 2.)
	
	return img[y:(y+dy), x:(x+dx), :]


def flip_axis(x, axis):

	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)

	return x

def horizontal_flip(x):

	if(np.random.random() < 0.5):
		x = flip_axis(x, 1)
	
	return x

def vertical_flip(x):

	if(np.random.random() < 0.5):
		x = flip_axis(x, 0)
	
	return x


def multiscale_crop(img, final_size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1, interpolation=pil_image.BILINEAR):

	assert img.shape[2] == 1 or img.shape[2] == 3

	height, width = img.shape[0], img.shape[1]
	
	#begin Fill crops sizes
	crop_sizes = []
	min_size = np.min((img.shape[0], img.shape[1]))
		
	for h in range(len(scale_ratios)):

		crop_h = int(min_size * scale_ratios[h])
		for w in range(len(scale_ratios)):

			crop_w = int(min_size * scale_ratios[w])
			crop_sizes.append((crop_h, crop_w))
	
	#end Fill crop sizes
	
	size_crop_selected = np.random.randint(0, len(crop_sizes) - 1)
	crop = (crop_sizes[size_crop_selected][0], crop_sizes[size_crop_selected][1])
	
	if(fix_crop):
	
		#begin Fill offsets
		h_off = int((img.shape[0] - final_size) / 4)
		w_off = int((img.shape[1] - final_size) / 4)

		offsets = []
		offsets.append((0, 0))          # upper left
		offsets.append((0, 4*w_off))    # upper right
		offsets.append((4*h_off, 0))    # lower left
		offsets.append((4*h_off, 4*w_off))  # lower right
		offsets.append((2*h_off, 2*w_off))  # center

		if more_fix_crop:
			offsets.append((0, 2*w_off))        # top center
			offsets.append((4*h_off, 2*w_off))  # bottom center
			offsets.append((2*h_off, 0))        # left center
			offsets.append((2*h_off, 4*w_off))  # right center

			offsets.append((1*h_off, 1*w_off))  # upper left quarter
			offsets.append((1*h_off, 3*w_off))  #if (np.absolute(h-w) <= self.max_distort): upper right quarter
			offsets.append((3*h_off, 1*w_off))  # lower left quarter
			offsets.append((3*h_off, 3*w_off))  # lower right quarter
		
		off_sel = np.random.randint(0, len(offsets)-1)
		h_off = offsets[off_sel][0]
		w_off = offsets[off_sel][1]
		#end Fill offsets
	
	else:
		h_off = np.random.randint(0, img.shape[0] - final_size)
		w_off = np.random.randint(0, img.shape[1] - final_size)
	
	crop_img = img[h_off:h_off+crop[0], w_off:w_off+crop[1], :]
	final_img = array_to_img(crop_img)
	final_img = final_img.resize((final_size, final_size), interpolation)
	final_img = img_to_array(final_img)
	#final_img = cv2.resize(crop_img, (final_size, final_size), interpolation)
	return final_img


def array_to_img(x, scale=True):

    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype="float32")

    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img):

    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype="float32")

    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    return x


def train_generator(batches, final_size, scale_ratios, channels, save_to_dir=None, save_format='.jpg'):

	while True:
		batch_x, batch_y = next(batches)
		batch_crops = np.zeros((batch_x.shape[0], final_size, final_size, channels))
		for i in range(batch_x.shape[0]):
	
			batch_crops[i] = multiscale_crop(batch_x[i], final_size, scale_ratios)
			#batch_crops[i] = center_crop(batch_x[i], final_size)			
			batch_crops[i] = horizontal_flip(batch_crops[i])
			
			if(save_to_dir):
				img = array_to_img(batch_crops[i], scale=True)
				fname = 'aug_{index}_{hash}.{format}'.format(index=i, hash=np.random.randint(1e7), format=save_format)
				img.save(os.path.join(save_to_dir, fname))				

		yield (batch_crops, batch_y)


def valid_generator(batches, final_size, channels, save_to_dir=None, save_format='.jpg'):

	while True:
		batch_x, batch_y = next(batches)
		batch_crops = np.zeros((batch_x.shape[0], final_size, final_size, channels))
		for i in range(batch_x.shape[0]):
			
			batch_crops[i] = center_crop(batch_x[i], final_size)
	
			if(save_to_dir):
				img = array_to_img(batch_crops[i], scale=True)
				fname = 'aug_{index}_{hash}.{format}'.format(index=i, hash=np.random.randint(1e7), format=save_format)
				img.save(os.path.join(save_to_dir, fname))	
		
		yield (batch_crops, batch_y)







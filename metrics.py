import os
import argparse
import numpy as np
import sys


def _parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("truth", help="File with the ground truth bounding boxes for a video")
	parser.add_argument("predicted", help="File with hte predicted bounding boxes for the same video")
	parser.add_argument("outdir", help="Directory the calculate metrics")
	
	
	return parser.parse_args()
	
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
	# Max between zero  and the difference is for the case where the intersection doesn't exist, ie, its empty and thus zero
    inter_area = max(xi2 - xi1, 0)*max(yi2 - yi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0])*(box1[3] - box1[1])
    box2_area = (box2[2] - box2[0])*(box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area/union_area
    
    return iou

def precision(true_positive, total_positive):
	"""Implement the precision metric - number of true positives divided by the total number of elements labeled as belonging to the positive class
    
    Arguments:
    true_positive -- number of true positive (i.e. the number of items correctly labeled as belonging to the positive class)
	total_positive -- total number of elements labeled as belonging to the positive class (i.e. the sum of true positives and false positives, which are items incorrectly labeled as belonging to the class)
    """

	return true_positive/total_positive
	
def recall(true_positive, real_positive):
	"""Implement the recall metric - number of true positives divided by the total number of elements that actually belong to the positive class
    
    Arguments:
    true_positive -- number of true positive (i.e. the number of items correctly labeled as belonging to the positive class)
	real_positive -- total number of elements that actually belong to the positive class (i.e. the sum of true positives and false negatives, which are items which were not labeled as belonging to the positive class but should have been)
    """

	return true_positive/real_positive

def fmeasure(prec, rec):

	return (2*prec*rec)/(prec + rec)
	
def sample_iou(ground_truth_file, predict_file, outdir, keep=1):
	""" Calculate and save the iou between bounding boxes of the ground truth and predicted values
	
	Arguments:
	ground_truth_file --  name of the file containing the ground truth values of bounding boxes for some frames of video
	predict_file --  name of the file containing the predicted values of bounding boxes for all frames of video
	outdir --  output directory to where the iou file will be writed
	keep -- how many subdirectories from the path of the `ground_truth_file` to keep when saving the output file in the `outdir`
	"""

	# read the file with the bounding boxes of a video and convert the information to numpy array of floats
	with open(ground_truth_file, 'r') as infile:
		# read all file, split by lines, then inside lines split by space, then cast every element to float, then create a numpy array with this
		truth = np.array([list(map(float, line.split())) for line in infile.read().splitlines()])
		# so every row will be a frame, and the collums have frame number, x1, y1, x2, y2
	# same above explanation applies to the following
	with open(predict_file, 'r') as infile:
		predict = np.array([list(map(float, line.split())) for line in infile.read().splitlines()])
		
	# when predict have more frame than truth...
	truth_frames = list(truth[:,0])
	# ...I get the iou only from those frames whose are present in them both
	frames_iou = ["{} {}".format(int(frame[0]), iou(frame[1:], truth[truth_frames.index(frame[0]),1:])) for frame in predict if frame[0] in truth_frames]
	
	# formating again one frame per line
	frames_iou = "\n".join(frames_iou)
	
	# join (outdir, the subdirectories to keep, filename with extension)
	path = os.path.join(outdir, *(ground_truth_file.split(os.sep)[-(keep+1):-1]))
	os.makedirs(path, exist_ok=True)
	name = os.path.splitext(os.path.basename(ground_truth_file))[0] + '.iou'
	name = os.path.join(path, name)
	
	with open(name, 'w') as outfile:
		outfile.write(frames_iou)
	
if __name__ == "__main__":

	args = _parse_args()
	sample_iou(args.truth, args.predicted, args.outdir)
	
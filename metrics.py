import os
import argparse
import numpy as np
import sys


def _parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory with dataset to be splited")
	parser.add_argument("split_train", help="Text file with the split for the train set")
	parser.add_argument("split_valid", help="Text file with the split for the validation set")
	parser.add_argument("-ext","--extension", help="Extension of files with dot", default=".jpg")
	parser.add_argument("-t","--type", help="Type of split file:\n0 - video_sample nb_frames class\n1 - Class_Folder/video_sample.ext", default=0, type=int)
	
	
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
	
if __name__ == "__main__":

	args = _parse_args()
	_main(args.dir, args.split_train, args.split_valid, args.extension)
	
import numpy as np
import os
import argparse
from pprint import pprint
from tqdm import tqdm
import json

def _get_Args():

	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", help="Directory containing the dataset validation set folder")

	return parser.parse_args()


def load_filenames(filepath):
    # list the classes
    total_samples = sum([len(files) for r, d, files in os.walk(filepath)])


    with tqdm(total=total_samples, ascii=True) as pbar:
        for idx, subdir in enumerate(sorted(os.listdir(filepath))):
            dirpath = os.path.join(filepath, subdir)
            # if it's a folder
            for img_file in sorted(os.listdir(dirpath)):
                name = os.path.join(dirpath, img_file)
                feat_map = np.load(name)
                # the correspondent class is the last class listed
                #preds.append(idx == np.argmax(feat_map))
                feat_map.T[[34, 35]] = feat_map.T[[35, 34]]
                feat_map.T[[47, 46]] = feat_map.T[[46, 47]]
                np.save(name, feat_map)

                pbar.update(1)

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	load_filenames(args.dataset)

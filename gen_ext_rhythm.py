import os
import argparse
import json
import shutil

from list_dir import list_dir, write_list
from parallelize_script import parallelize
from SplitScript import _main as split_files


def _get_Args():

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("dir",
				help = "Directory with dataset")
	parser.add_argument("outdir",
				help = "Output directory with dataset")
	parser.add_argument("split_folder",
				help="Directory contaning split files")
	parser.add_argument("-sfm","--split_file_mask",
				default="{set}list{:02d}.txt",
				help="String mask for names of split files")
	parser.add_argument("-db", "--database_name",
				help= "Database name")
	parser.add_argument("-e", "--ext", default=".avi",
				help= "Extension of output")
	parser.add_argument("-m", "--mode", default="mean",
				help= "Capture mode", choices=['mean', 'gaussian'])
	parser.add_argument("-sg", "--sigma", type=float,
				help= "Standard deviation to use whitin gaussian filter")
	parser.add_argument("-sz", "--size", type=float,
				help= "Filter size")
	parser.add_argument("-d", "--direction", default='H',
				help= "Visual rhythm direction", choices=["V", "H", "v", "h"])
	parser.add_argument("-p", "--percentil", type=float, default=[0.5],
				help= "Relative position to extract the rhythm", nargs='+')
	parser.add_argument("-cm", "--color_mode", default='rgb',
				help= "Color mode", choices=['rgb', 'gray', 'ic'])
	parser.add_argument("-fs", "--frame_mask", type=int, default=1,
				help = "Sample image by picking frames in jumps of this value")
	parser.add_argument("-ts", "--target_size", type=int, default=224,
				help = "Final image size (target_size x target_size)")
	parser.add_argument("-n", "--num", type=int, default=1,
				help = "Number os samples to create without overlaping")
	parser.add_argument("-c", "--crop", type=int, default=1,
				help = "How many vertical crops to take", choices=[1,2,3])
	parser.add_argument("-s", "--stride", type=int, default=0,
				help = "Frames to jump when overlapping the windows")

	args = parser.parse_args()

	if args.mode == 'gaussian' and args.size is None and args.sigma is None:
		parser.error("--mode gaussian requires --size and --sigma to be set.")

	return args

def get_db_name(path):

	db_names = ['ucf101', 'ucf11', 'hmdb51', 'hmdb']
	for name in db_names:
		if name in (path.lower()):
			return name
	raise ValueError('Unable to get database name! ' +
	'Run the script again and provide database name as parameter.')

def split_rhythms(db, rhythm_dir, rhythm_split_dir, split_folder, file_mask, ext, data):

	db = db.lower()
	if db == 'ucf101':
		for i in range(1,4):
			train_folder = os.path.join(split_folder, file_mask.format(i, set='train'))
			valid_folder = os.path.join(split_folder, file_mask.format(i, set='test'))
			split_files(rhythm_dir, train_folder, valid_folder, ext, 1, split=i, outdir=rhythm_split_dir)
	elif db == 'ucf11':
		for i in range(1,4):
			train_folder = os.path.join(split_folder, file_mask.format(i, set='train'))
			valid_folder = os.path.join(split_folder, file_mask.format(i, set='test'))
			split_files(rhythm_dir, train_folder, valid_folder, ext, 1, split=i, outdir=rhythm_split_dir)
	elif db in ['hmdb', 'hmdb51']:
			split_files(rhythm_dir, split_folder, split_folder, ext, 2, outdir=rhythm_split_dir)

	data = data.copy()
	del data['output']
	for i in range(1,4):
		data['split'] = i
		filepath = os.path.join(rhythm_split_dir, os.path.split(rhythm_dir)[-1] + "_split{}".format(i), 'info.json')
		with open(filepath, 'w') as fp:
			json.dump(data, fp, sort_keys=True, indent=4)

def split_augmented_rhythms(db, rhythm_dir, rhythm_split_dir, split_folder, file_mask, ext, data, aug_factor):

	db = db.lower()
	if db == 'ucf101':
		for i in range(1,4):
			train_folder = os.path.join(split_folder, file_mask.format(i, set='train'))
			valid_folder = os.path.join(split_folder, file_mask.format(i, set='test'))
			split_files(rhythm_dir, train_folder, valid_folder, ext, 1, split=i, aug_factor=aug_factor, outdir=rhythm_split_dir)
	elif db == 'ucf11':
		for i in range(1,4):
			train_folder = os.path.join(split_folder, file_mask.format(i, set='train'))
			valid_folder = os.path.join(split_folder, file_mask.format(i, set='test'))
			split_files(rhythm_dir, train_folder, valid_folder, ext, 1, split=i, aug_factor=aug_factor, outdir=rhythm_split_dir)
	elif db in ['hmdb', 'hmdb51']:
			split_files(rhythm_dir, split_folder, split_folder, ext, 2, aug_factor=aug_factor, outdir=rhythm_split_dir)

	data = data.copy()
	del data['output']
	data['aug_factor'] = aug_factor
	for i in range(1,4):
		data['split'] = i
		filepath = os.path.join(rhythm_split_dir, os.path.split(rhythm_dir)[-1] + "_split{}".format(i), 'info.json')
		with open(filepath, 'w') as fp:
			json.dump(data, fp, sort_keys=True, indent=4)

def create_rhythms(db_name, path, mode, color_mode, direction, size, sigma,
				   percentil, ext, outdir, split_folder, split_file_mask):

	db_name = db_name or get_db_name(path)
	temp_dir = "temp_dir"

	if mode == 'gaussian':
		rhythm_folder = "{}_VR_{}_{}_Gaussian_SZ_{}_SG_{}_P_{}".format(db_name,
		color_mode, direction, size, sigma, percentil).upper()

	else:
		rhythm_folder = "{}_VR_{}_{}_Mean_SZ_{}_SG_{}_P_{}".format(db_name,
		color_mode, direction, size, sigma, percentil).upper()

	rhythm_dir =  os.path.join(temp_dir, rhythm_folder)
	# listar diretorios
	files = list_dir(path, ext)
	# salvar listas
	write_list(files, "temp_videos_list_{}.txt".format(db_name), temp_dir)
	vid_list_file = os.path.join(temp_dir, "temp_videos_list_{}.txt".format(db_name))
	# criar arquivo de parametros para o ritmo
	data = {
		'output': temp_dir,
		'ext': '.jpg',
		'mode': mode,
		'sigma': sigma or 0.0,
		'size': size or 0.0,
		'direction': direction,
		'percentil': ' '.join(map(str, percentil)),
		'color': color_mode,
		'db_name': db_name
	}

	parameters = ("{output} -e {ext} -m {mode} -sg {sigma} -s {size} " +
				  "-d {direction} -p {percentil} -c {color} -db " +
				  "{db_name} -kds 1").format(**data)

	params_file = os.path.join(temp_dir,
					"temp_params_rhythm_{}.txt".format(db_name))

	with open(params_file, "w") as f:
		f.write(parameters)

	# criar ritmos paralelamente
	parallelize("VideoCapture.py", [vid_list_file], params_file, 1)
	# separar ritmos nos splits
	# criar txt com as informações de cada split e salvar nas pastas de cada split
	rhythm_split_dir = os.path.join(outdir, rhythm_folder)
	split_rhythms(db_name, rhythm_dir, rhythm_split_dir, split_folder,
				  split_file_mask, '.jpg', data)

	return rhythm_folder, rhythm_dir, data

def extend_rhythms(db_name, rhythm_folder, rhythm_dir, data, num, crop, stride,
				   target_size, frame_mask, outdir, split_folder,
				   split_file_mask):

	db_name = db_name or get_db_name(path)
	temp_dir = "temp_dir"

	# listar diretorio dos ritmos criados
	files = list_dir(rhythm_dir, '.jpg')
	# salvar lists dos ritmos criados
	write_list(files, "temp_rhythms_list_{}.txt".format(db_name), temp_dir)
	rhythm_list_file = os.path.join(temp_dir, "temp_rhythms_list_{}.txt".format(
									db_name))

	rhythm_da_folder = rhythm_folder + ("_WINDOW_{}_CROP_{}_STRIDE_{}_"+
							"TS_{}").format(num, crop, stride, target_size)

	rhythm_da_dir = os.path.join(temp_dir, rhythm_da_folder)

	# criar arquivo de parametros para o aumento do ritmo
	data2 = {
		'output': rhythm_da_dir,
		'frame_mask': frame_mask,
		'target_size': target_size,
		'num': num,
		'crop': crop,
		'stride': stride
	}

	parameters = ("{output} -kds 1 -fs {frame_mask} -ts {target_size} "+
					"-n {num} -c {crop} -s {stride}").format(**data2)

	params_file = os.path.join(temp_dir,
					"temp_params_da_{}.txt".format(db_name))

	with open(params_file, "w") as f:
		f.write(parameters)

	# aumentar ritmos paralelamente
	parallelize("rhythm_da.py", [rhythm_list_file], params_file, 1)
	# separar ritmos aumentados nos splits
	# criar txt com as informações de cada split e salvar nas pastas de cada split
	rhythm_da_split_dir = os.path.join(outdir, rhythm_da_folder)
	split_augmented_rhythms(db_name, rhythm_da_dir, rhythm_da_split_dir,
		split_folder, split_file_mask, '.jpg', {**data, **data2},
		frame_mask*num*crop)

	print("Removing temporary files...")
	# remover arquivos dos ritmos
	# remover arquivos dos ritmos estendidos
	shutil.rmtree(temp_dir)


def _main(args):

	rhythm_folder, rhythm_dir, data = create_rhythms(args.database_name,
				   args.dir, args.mode, args.color_mode, args.direction,
				   args.size, args.sigma, args.percentil, args.ext,
				   args.outdir, args.split_folder, args.split_file_mask)

	extend_rhythms(args.database_name, rhythm_folder, rhythm_dir, data,
				   args.num, args.crop, args.stride, args.target_size,
				   args.frame_mask, args.outdir, args.split_folder,
				   args.split_file_mask)

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)

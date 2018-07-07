import argparse
import os

encode = "utf-8"

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory to list")
	parser.add_argument("name", help="Output file name")
	parser.add_argument("ext", help="File extension to filter")
	parser.add_argument("-o", "--outdir", help="Output directory", nargs='?', const='', default='')
	return parser.parse_args()
	
	
def list_dir(dir, ext):
	
	files_list = []
	if os.path.isdir(dir):
		for root, dirs, files in os.walk(dir):
			files_list += [os.path.join(root, file) for file in files if os.path.splitext(file)[1] == ext]
		
	else:
		print("Diretorio invalido!!")
	
	return files_list
	
def write_list(files, name, outdir):
	
	if not os.path.isdir(outdir) and outdir != '':
		os.makedirs(outdir)
	
	out_file = os.path.join(outdir, name)
	
	with open(out_file, "w", encoding = encode) as output:
		output.write('\n'.join(files))
	
def _main(args):

	files = list_dir(args.dir, args.ext)
	write_list(files, args.name, args.outdir)
	
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)

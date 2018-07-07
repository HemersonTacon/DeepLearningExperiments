import multiprocessing as mp
import subprocess as sp
import argparse
import time
from tqdm import tqdm

encode = "utf-8"
threads_file = 'threads.txt'

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("script", help="Script name to run in parallel")
	parser.add_argument("files", help="Name of the file with the names of the input files for the script", nargs='+')
	parser.add_argument("params", help="Name of the file with the extra parameters of script")
	parser.add_argument("-c","--charge", type=int, help="This value multiplied with number of threads avaiable indicate how many process will run between reads of the file with the number of threads")
	return parser.parse_args()

def get_threads():

	try:
		with open(threads_file, "r", encoding=encode) as input:
			# Reading number of threads
			threads = int(input.read())
			return threads
	except Exception as e:
		print("A problem ocurred trying to read the number of threads: ", e)
		
def get_files(files):

	try:
		all_names = []
		for file in files:
			with open(file, "r", encoding=encode) as input:
				# Reading names of input files to the script
				names = input.read().splitlines()
				all_names.append(names)
		# transpose matrix to get the right 'pairs'
		return list(zip(*all_names))
	except Exception as e:
		print("A problem ocurred trying to read the file with the names of inputs: ", e)
		return 1
		
def get_params(params_file):

	try:
		with open(params_file, "r", encoding=encode) as input:
			# Reading parameters of script (all in one line)
			params = input.readline().split()
			return params
	except Exception as e:
		print("A problem ocurred trying to read the file with the parameters: ", e)
		return 1
	
def parallelize(script, files, params, charge_factor):
	
	if charge_factor == None:
		charge_factor = 10
	
	names = get_files(files)
	params = get_params(params)
	# Number of scripts already executed
	executions = 0
	# Maximun number of scripts executions
	max_executions = len(names)
	
	with tqdm(total=max_executions, ascii=True) as pbar:
		while executions < max_executions:
			
			num_threads = get_threads()
			executions_interval = min(charge_factor*num_threads, max_executions - executions)
			configurations = []
			#print(time.ctime(), " Executing task ",executions," to ", executions+executions_interval, " of ", max_executions)
			
			for i in range(executions_interval):
				command = list(params)
				[command.insert(0, name) for name in reversed(names[executions])]
				command.insert(0, script)
				command.insert(0, 'python')
				configurations.append(command)
				executions = executions + 1
				
			pool = mp.Pool(processes=num_threads)
			pool.map(sp.call, configurations)
			pool.close()
			pbar.update(executions_interval)
	
			
	
def _main(args):
	
	parallelize(args.script, args.files, args.params, args.charge)
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)

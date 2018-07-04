import os
import shutil

experiment_number_file = "experiment.txt"

## TODO: ADICIONAR NO ARQUIVO DE TEXTO O SCORE OBTIDO NO CONJUNTO DE TESTE, COLOCAR ESSE PARAMETRO DE SCORE PRA SER PASSADO TBM NESSA FUNCAO

def save_infos(nm_script, args, history, best_epoch, test_score, nm_weights, nm_weights_best, nm_plot_acc, nm_plot_loss, time_info, dir, file_name = "Experimento"):
	
	try:
		with open(experiment_number_file, "r") as input:
			# Reading number of experiments
			nb_xp = int(input.read())
	except Exception as e:
		print("A problem ocurred trying to read the number of experiments in file ",experiment_number_file,": ", e)
		exit(1)
	
	# Concatenating infos in a single string to store in a file later
	info = ""
	
	info+= "{:*^50}\n\n".format(" About Time ")
	info+= "  Begin time: {}\n".format(time_info[0])
	info+= "    End time: {}\n".format(time_info[1])
	info+= "Elapsed time: {}\n\n".format(time_info[2])
	
	info+= "{:*^50}\n\n".format(" About Script and Parameters ")
	info+= "Executed {} script with following parameters:\n".format(nm_script)
	info+= "{}\n\n".format(str(vars(args)))
	
	info+= "{:*^50}\n\n".format(" About Created Files ")
	info+= "              File with model weights: {} \n".format(nm_weights)
	info+= "File with model weights of best epoch: {} \n".format(nm_weights_best)
	info+= "   Image plot of accuracy over epochs: {} \n".format(nm_plot_acc)
	info+= "       Image plot of loss over epochs: {} \n\n".format(nm_plot_loss)
	
	info+= "{:*^50}\n\n".format(" About Accuracy and Loss ")
	info+= "Best accuracy (epoch {}): \nloss: {:6.4f} acc: {:6.4f} val_loss: {:6.4f} val_acc: {:6.4f} \n".format(best_epoch+1, history.history['loss'][best_epoch], history.history['acc'][best_epoch], history.history['val_loss'][best_epoch], history.history['val_acc'][best_epoch])
	info+= "Accuracy and loss of test set: {}\n".format(test_score)
	info+= "History of accuracy and loss: \n"
	epochs = len(history.history['loss'])
	
	for i in range(epochs):
		info+= "Epoch {}/{}\nloss: {:6.4f} acc: {:6.4f} val_loss: {:6.4f} val_acc: {:6.4f}\n".format(i+1, epochs, history.history['loss'][i], history.history['acc'][i], history.history['val_loss'][i], history.history['val_acc'][i])
	

	out_dir = os.path.join(dir,file_name+str(nb_xp))
	os.mkdir(out_dir)
	log_file = os.path.join(out_dir, "log.txt")
	
	with open(log_file, "w") as output:
		output.write(info)
	
	# copying created files
	shutil.copy(nm_weights, out_dir)
	shutil.copy(nm_weights_best, out_dir)
	shutil.copy(os.path.join("imgs", nm_plot_acc), out_dir)
	shutil.copy(os.path.join("imgs", nm_plot_loss), out_dir)
	
	with open(experiment_number_file, "w") as output:
		# Updating number of experiments
		nb_xp+=1
		output.write(str(nb_xp))
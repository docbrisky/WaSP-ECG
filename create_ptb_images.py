import multiprocessing
from ptb_loader import Physionet_Helper
import os
from config import config
from helper_functions import n_classes
from image_augmentor import Image_Augmentor
import numpy as np
import shutil

def reset_PTB():

	if os.path.isdir(config.ptb_folder):
		shutil.rmtree(config.ptb_folder)
	if os.path.isdir(config.ptb_val_folder):
		shutil.rmtree(config.ptb_val_folder)
	if os.path.isdir(config.ptb_test_folder):
		shutil.rmtree(config.ptb_test_folder)
	if os.path.isdir('pickles'):
		shutil.rmtree('pickles')
	create_ptb_images()

def multiprocess_function(args):

	pt_id = args[1]

	X, y = args[0].load_sample_ecg(args[1])

	X = args[2].transform_image(X)

	np.save(args[3] + '/' + str(pt_id[0]) + '_ECG.npy', X)
	np.save(args[3] + '/' + str(pt_id[0]) + '_label.npy', y)

	print('Written ECG ' + str(args[1][0]), end = '\r')

def create_ptb_images():

	folders = [config.ptb_folder, config.ptb_val_folder, config.ptb_test_folder]

	if os.path.isdir(config.ptb_folder) and \
		os.path.isdir(config.ptb_val_folder) and \
		os.path.isdir(config.ptb_test_folder):

		print('PTB folders already exist')

	else:

		img_aug = Image_Augmentor(
			dim = config.dim, 
			n_classes = n_classes, 
			augment = False,
			waves_only = config.waves_only
			)

		ph = Physionet_Helper()
		train_list = ph.get_train_list()
		val_list = ph.get_val_list()
		test_list = ph.get_test_list()
		arg_list = []
		sample_lists = [train_list, val_list, test_list]

		for i in range(3):

			folder = folders[i]
			sample_list = sample_lists[i]

			if not os.path.isdir(folder):

				os.mkdir(folder)

			for s in sample_list:

				arg_list.append([ph, s, img_aug, folder])

			pool = multiprocessing.Pool(processes = config.cpu_cores)
			pool.map(multiprocess_function, arg_list)

		print('\nWritten all PTB files')

if __name__ == "__main__":

	create_ptb_images()
# IMPORT DEPENDENCIES

import glob
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import JaccardLoss
from monai.losses import DiceLoss
from monai.metrics import ConfusionMatrixMetric
import os
import random
import cv2
import platform
import time
import multiprocessing
from config import config
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import class_colors, n_classes, points, waves, segmentation_key
from mpi4py import MPI
from image_augmentor import Image_Augmentor
from training_functions import DataGenerator, \
							multiprocess_generator
random.seed(config.random_seed)

##############################################################

def generate_data():

	# LOAD CONFIG PARAMETERS
	cpu_cores = config.cpu_cores
	augment = config.augment
	p_multiplier = config.p_multiplier
	t_multiplier = config.t_multiplier
	dim = config.dim
	noise_multiplier = config.noise_multiplier
	impedance = config.impedance
	rhythms = config.rhythms
	folder = config.folder
	ECG_suffix = config.ECG_suffix
	label_suffix = config.label_suffix
	use_background = config.use_background
	background_folder = config.background_folder
	cpu_cores = config.cpu_cores
	use_multiprocessing = config.use_multiprocessing

	##############################################################

	# SET UP THE FRAMEWORKS AND DATA LOADERS / GENERATORS:

	# prepare saving / loading functions
	# (os.path.join wasn't used in original script):
	if platform.system() == 'Windows':
		splitter = '\\'
	else:
		splitter = '/'

	# set up the multiprocessing pool if appropriate:
	if use_multiprocessing:
		pool = multiprocessing.Pool(processes = cpu_cores)

	existing_files = []
	if not os.path.exists(folder):
		os.makedirs(folder)
	existing_files += glob.glob(folder + splitter + '*.npy')

	print('Existing files:', len(existing_files), 'in', folder)

	##############################################################

	# GENERATE NEW DATA:

	if len(existing_files) == 0:

		print('Generating data...')

		cpu_cores = cpu_cores

		if len(existing_files) == 0:
			counter = 0
		else:
			test_numbers = []
			for t in existing_files:
				test_numbers.append(int(t.split(splitter)[-1].split('_')[0]))
			counter = max(test_numbers) + 1

		# load list of background images (MSCOCO 2017 validation images used in original implementation):
		if use_background:
			bg_image_list = glob.glob(background_folder)
		else:
			bg_image_list = None

		# create training data generator:
		train_gen = DataGenerator(
				folder = folder, 
				dim = config.save_dim, 
				ECG_suffix = ECG_suffix, 
				label_suffix = label_suffix, 
				augment = False, 
				rhythms = rhythms, 
				impedance = impedance, 
				noise_multiplier = noise_multiplier, 
				counter = counter, 
				rates = config.rates, 
				bg_image_list = bg_image_list, 
				p_multiplier = p_multiplier, 
				t_multiplier = t_multiplier, 
				save_data = True,
				waves_only = config.waves_only,
				one_dimensional = config.one_dimensional
				)

		for h in range(config.generation_cycles):

			start = time.perf_counter()

			# generate data:
			if use_multiprocessing:
				pool_list = []
				for i in range(cpu_cores):
					pool_list.append([train_gen, counter + i])

				_ = pool.map(multiprocess_generator, pool_list)

			else:
				for i in range(cpu_cores):
					_, _ = train_gen.__getitem__(counter + i)

			counter += cpu_cores

			print('\nBatch', h, 'of', config.generation_cycles, 'written.', \
			 	'That took  {:.2f}'.format(time.perf_counter() - start), 'seconds', end = '\r')

		print('Done!')

	else:

		print('Data already exists at', folder)

if __name__ == '__main__':

	generate_data()
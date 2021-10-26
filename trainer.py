# IMPORT DEPENDENCIES

import glob
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import JaccardLoss
from monai.losses import DiceLoss
from monai.metrics import ConfusionMatrixMetric
from unet_1d import UNET_1D, Classification_Head, PTB_Model
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
from training_functions import Classifier, \
								Classifier_ptb, \
								DS_Loader, \
								Physionet_Loader, \
								multitask_loop, \
								single_task_loop

##############################################################

def train(
	file_suffix = '2d_pretrain',
	model_to_save = [config.model + '.p', ''],
	model_to_load = [config.model + '.p', '', ''],
	training_task = 'segmentation',
	learning_rates = [1e-4, 1e-6],
	buffer_size = 64,
	n_outputs = 7,
	ptb_outputs = 2,
	ptb_classes = 2,
	train_model_per_class = False,
	mix_dimensions = False,
	completely_naive = False
	):

	one_dimensional = config.one_dimensional
	
	if training_task == 'classification':
		folder = config.ptb_folder
	else:
		folder = config.folder

	if train_model_per_class:
		ptb_outputs = 1

	if one_dimensional:
		mix_dimensions = False

	random.seed(config.random_seed)

	# LOAD CONFIG PARAMETERS
	batch_size = config.batch_size
	ECG_suffix = config.ECG_suffix
	label_suffix = config.label_suffix
	cpu_cores = config.cpu_cores
	use_deepspeed = config.deepspeed
	use_half = config.use_half
	epochs = config.epochs
	use_multiprocessing = config.use_multiprocessing

	if one_dimensional and use_deepspeed:
		print('Error, DeepSpeed not currently supported for one dimensional models.\n' + \
			'Please set to \'False\' in config file.')
		quit()

	# prepare saving / loading functions
	# (os.path.join wasn't used in original script):
	if platform.system() == 'Windows':
		splitter = '\\'
	else:
		splitter = '/'

	if not os.path.isdir(config.results_folder):
		os.mkdir(config.results_folder)
	if not os.path.isdir(config.model_folder):
		os.mkdir(config.model_folder)

	##############################################################

	# PREPARE DATA & MODEL:

	# load training data list:
	existing_files = []
	existing_files += glob.glob(folder + splitter + '*.npy')
	if len(existing_files) == 0:
		print('Error, no files found!' \
				+ ' Set \"train = False\" in config.py and re-run this script to generate data.')
		quit()

	# set buffer size:
	buffer_size = config.buffer_size

	# edit parameters if using DeepSpeed:
	if use_deepspeed:
		batch_size = config.gpus * batch_size
		use_multiprocessing = False

	# set CUDA device:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# set loss functions:
	if training_task == 'multitask':
		criterion = torch.nn.BCELoss()
		seg_criterion = DiceLoss(include_background = True, jaccard = True)
		f1 = ConfusionMatrixMetric(metric_name = 'f1_score', include_background = True)
	elif training_task == 'classification':
		criterion = torch.nn.BCELoss()
		seg_criterion = None
		f1 = None
	elif training_task == 'segmentation':
		criterion = DiceLoss(include_background = True, jaccard = True)
		seg_criterion = None
		f1 = ConfusionMatrixMetric(metric_name = 'f1_score', include_background = True)
	else:
		print('Error, training task ' + training_task + ' not recognised!')
		quit()

	# set CSV file for logging:
	csv = '\nEpoch, Loss, F1\n'

	if one_dimensional:
		# create the 1D U-Net
		model = UNET_1D(n_classes) 
	else:
		if completely_naive:
			# create the 2D U-Net
			model = smp.Unet(
				encoder_name = config.model,
				encoder_weights = None,
				in_channels = 3,
				classes = n_classes,
				decoder_attention_type = 'scse',
				activation = 'softmax'
				)
		else:
			# create the 2D U-Net
			model = smp.Unet(
				encoder_name = config.model,
				encoder_weights = config.encoder_weights,
				in_channels = 3,
				classes = n_classes,
				decoder_attention_type = 'scse',
				activation = 'softmax'
				)

	if os.path.isfile(config.model_folder + splitter + model_to_load[0]):
		model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[0]))
		print('Loaded previous weights')
	
	optimizer_enc = None
	encoder = None

	if training_task == 'multitask':
		if one_dimensional:
			encoder = model
			model = Classification_Head(encoder = encoder, n_classes = n_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[1]):
				model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[1]))
				print('Loaded previous weights for classification head')
			encoder_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
			optimizer_enc = torch.optim.Adam(encoder_parameters, lr = learning_rates[1])
		else:
			model = Classifier(model, output_classes = n_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[1]):
				model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[1]))
				print('Loaded previous weights for classification head')
	elif training_task == 'classification':
		if one_dimensional:
			encoder = model
			head = Classification_Head(encoder = encoder, n_classes = n_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[1]):
				head.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[1]))
			model = PTB_Model(encoder, head, n_classes = ptb_outputs)
		elif mix_dimensions:
			encoder = model
			base_1d_model = UNET_1D(
				n_classes
				)
			if os.path.isfile(config.model_folder + splitter + model_to_load[1]):
				base_1d_model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[1]))
				print('Loaded previous weights for 1D model')
			head = Classification_Head(encoder = base_1d_model, n_classes = n_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[2]):
				head.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[2]))
			model = PTB_Model(base_1d_model, head, n_classes = ptb_outputs)
		else:
			model = Classifier(model, output_classes = n_outputs, penultimate = True)
			if os.path.isfile(config.model_folder + splitter + model_to_load[1]):
				model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[1]))
				print('Loaded previous weights for classification head')
			model = Classifier_ptb(model, output_classes = ptb_outputs)
	elif training_task != 'segmentation':
		print('Error - training task ' + training_task + ' not recognised!')
		quit()

	parameters = filter(lambda p: p.requires_grad, model.parameters())

	if use_deepspeed:
		import deepspeed
		from deepspeed_argparser import add_argument
		from save_ds_as_torch import save_deepspeed
		args = add_argument()
		print('Using DeepSpeed framework')
	else:
		optimizer = torch.optim.Adam(parameters, lr = learning_rates[0])
		model = model.to(device)
		if encoder is not None:
			encoder = encoder.to(device)

	samples = glob.glob(folder + splitter + '*_' + ECG_suffix + '.npy')
	print('Training on', len(samples), 'samples')
	sample_numbers = [s.split(splitter)[-1].split('_')[0] for s in samples]

	ptb_switch = False
	if training_task == 'classification':
		ptb_switch = True

	##############################################################

	# RUN TRAINING

	# set up Torch dataset:
	if one_dimensional and training_task == 'classification':
		loader = Physionet_Loader(balance = True)
	else:
		loader = DS_Loader(
			sample_numbers,
			one_dimensional = one_dimensional,
			ptb = ptb_switch
			)

	if use_deepspeed:
		# initiate DeepSpeed engine:
		model_engine, optimizer, train_loader, _ = deepspeed.initialize(
			args = args,
			model = model,
			model_parameters = parameters,
			training_data = loader
			)

		if encoder is not None:
			encoder = encoder.to(model_engine.local_rank)

		model = None

	else:
		# initiate Torch dataloader:
		train_loader = torch.utils.data.DataLoader(
			loader, 
			batch_size = batch_size, 
			shuffle = True, 
			num_workers = cpu_cores,
			pin_memory = torch.cuda.is_available()
			)

		model_engine = None

	total_samples = len(train_loader)
	loss_list = []
	f1_list = []

	# run the training loop:
	for epch in range(epochs):

		if training_task == 'multitask':

			models, f1_list, loss_list = multitask_loop(train_loader,
											criterion,
											seg_criterion,
											optimizer,
											optimizer_enc,
											device=device,
											one_dimensional = one_dimensional,
											buffer_size = buffer_size,
											total_samples = total_samples,
											use_deepspeed = use_deepspeed,
											model = model,
											encoder = encoder,
											model_engine = model_engine,
											f1_list = f1_list,
											loss_list = loss_list
											)

			for mdl in range(len(models)):
				# save models at the end of each epoch:
				if use_deepspeed:
					save_deepspeed(models[mdl], config.model_folder + splitter + model_to_save[mdl])
				else:
					torch.save(models[mdl].state_dict(), config.model_folder + splitter + model_to_save[mdl])

			# SAVE METRICS:
			csv = ''
			for i in range(len(loss_list)):
				csv += str(loss_list[i]) + '\n'
			with open(config.results_folder + splitter + \
					'loss_' + file_suffix + '_epoch_' + str(epch) + '.csv', 'w') as f:
				f.write(csv)
			csv = ''
			if len(f1_list) > 0:
				for i in range(len(f1_list)):
					csv += str(f1_list[i]) + '\n'
				with open(config.results_folder + splitter + \
						'f1_' + file_suffix + '_epoch_' + str(epch) + '.csv', 'w') as f:
					f.write(csv)

		else:

			if train_model_per_class:

				for tgt in range(ptb_classes):

					print('Training on target class', tgt)

					models, f1_list, loss_list = single_task_loop(
													train_loader,
													criterion,
													optimizer,
													device=device,
													buffer_size = buffer_size,
													total_samples = total_samples,
													use_deepspeed = use_deepspeed,
													model = model,
													encoder = encoder,
													model_engine = model_engine,
													f1_list = f1_list,
													loss_list = loss_list,
													training_task = training_task,
													target_class = tgt,
													one_dimensional = one_dimensional,
													mix_dimensions = mix_dimensions
													)

					for mdl in range(len(models)):
						# save models at the end of each epoch:
						save_path = config.model_folder + splitter + model_to_save[mdl] .split('.p')[0]
						save_path += '_' + str(tgt) + '.pth'
						if use_deepspeed:
							save_deepspeed(models[mdl], model_to_save[mdl])
						else:
							torch.save(models[mdl].state_dict(), model_to_save[mdl])

				# SAVE METRICS:
				csv = ''
				for i in range(len(loss_list)):
					csv += str(loss_list[i]) + '\n'
				with open(config.results_folder + splitter + \
						'loss_' + file_suffix + '_epoch_' + str(epch) + '_' + str(tgt) + '.csv', \
						'w') as f:
					f.write(csv)
				csv = ''
				if len(f1_list) > 0:
					for i in range(len(f1_list)):
						csv += str(f1_list[i]) + '\n'
					with open(config.results_folder + splitter + \
							'f1_' + file_suffix + '_epoch_' + str(epch) + '_' + str(tgt) + '.csv', 'w') as f:
						f.write(csv)

			else:

				models, f1_list, loss_list = single_task_loop(
													train_loader,
													criterion,
													optimizer,
													device=device,
													buffer_size = buffer_size,
													total_samples = total_samples,
													use_deepspeed = use_deepspeed,
													model = model,
													encoder = encoder,
													model_engine = model_engine,
													f1_list = f1_list,
													loss_list = loss_list,
													training_task = training_task,
													target_class = -1,
													one_dimensional = one_dimensional,
													mix_dimensions = mix_dimensions
													)

				for mdl in range(len(models)):
					# save models at the end of each epoch:
					if use_deepspeed:
						save_deepspeed(models[mdl], config.model_folder + splitter + model_to_save[mdl])
					else:
						torch.save(models[mdl].state_dict(), config.model_folder + splitter + model_to_save[mdl])

				# SAVE METRICS:
				csv = ''
				for i in range(len(loss_list)):
					csv += str(loss_list[i]) + '\n'
				with open(config.results_folder + splitter + \
						'loss_' + file_suffix + '_epoch_' + str(epch) + '.csv', 'w') as f:
					f.write(csv)
				csv = ''
				if len(f1_list) > 0:
					for i in range(len(f1_list)):
						csv += str(f1_list[i]) + '\n'
					with open(config.results_folder + splitter + \
							'f1_' + file_suffix + '_epoch_' + str(epch) + '.csv', 'w') as f:
						f.write(csv)


##############################################################
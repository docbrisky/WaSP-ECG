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
from rule_based_AF_detection import brute_force_search, predict_ecg
from training_functions import Classifier, \
								Classifier_ptb, \
								DS_Loader, \
								Physionet_Loader, \
								multitask_loop, \
								seg_mask_to_signal, \
								single_task_loop

##############################################################

def validate(
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
	test = False,
	completely_naive = False
	):

	one_dimensional = config.one_dimensional

	if test:
		folder = config.ptb_test_folder
	else:
		folder = config.ptb_val_folder

	if train_model_per_class:
		ptb_outputs = 1

	if training_task == 'rule_based':
		one_dimensional = False

	if one_dimensional:
		mix_dimensions = False

	random.seed(config.random_seed)

	# LOAD CONFIG PARAMETERS
	batch_size = config.batch_size
	ECG_suffix = config.ECG_suffix
	label_suffix = config.label_suffix
	cpu_cores = config.cpu_cores
	epochs = config.epochs
	use_multiprocessing = config.use_multiprocessing

	if one_dimensional:
		mix_dimensions = False

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

	# set CUDA device:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

	if mix_dimensions or training_task == 'rule_based':
		if os.path.isfile(config.model_folder + splitter + model_to_load[0]):
			model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[0]))
			print('Loaded previous weights for U-Net model')

	encoder = None

	if training_task == 'classification':
		if one_dimensional:
			encoder = model
			head = Classification_Head(encoder = encoder, n_classes = n_outputs)
			model = PTB_Model(encoder, head, n_classes = ptb_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[0]):
				model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[0]))
				print('Loaded previous weights for 1D classification head')
		elif mix_dimensions:
			encoder = model
			base_1d_model = UNET_1D(
				n_classes
				)
			head = Classification_Head(encoder = base_1d_model, n_classes = n_outputs)
			model = PTB_Model(base_1d_model, head, n_classes = ptb_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[1]):
				model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[1]))
				print('Loaded previous weights for 1D classification head')
		else:
			model = Classifier(model, output_classes = n_outputs, penultimate = True)
			model = Classifier_ptb(model, output_classes = ptb_outputs)
			if os.path.isfile(config.model_folder + splitter + model_to_load[0]):
				model.load_state_dict(torch.load(config.model_folder + splitter + model_to_load[0]))
				print('Loaded previous weights for 2D classification head')
	elif training_task != 'rule_based':
		print('Error - quantitative validation pipelines are only enabled for diagnostic classification')
		quit()

	model = model.to(device)
	if encoder is not None:
		encoder = encoder.to(device)

	samples = glob.glob(folder + splitter + '*_' + ECG_suffix + '.npy')
	if test:
		print('Testing on', len(samples), 'samples')
	else:
		print('Validating on', len(samples), 'samples')
	sample_numbers = [s.split(splitter)[-1].split('_')[0] for s in samples]

	##############################################################

	# RUN TRAINING

	# set up Torch dataset:
	if one_dimensional and training_task == 'classification':
		if test:
			loader = Physionet_Loader(test = True)
		else:
			loader = Physionet_Loader(val = True)
	else:
		if test:
			loader = DS_Loader(
				sample_numbers,
				one_dimensional = one_dimensional,
				ptb = True,
				ptb_test = True
				)
		else:
			loader = DS_Loader(
				sample_numbers,
				one_dimensional = one_dimensional,
				ptb = True,
				ptb_val = True
				)

	# initiate Torch dataloader:
	train_loader = torch.utils.data.DataLoader(
		loader, 
		batch_size = batch_size, 
		shuffle = True, 
		num_workers = cpu_cores,
		pin_memory = torch.cuda.is_available()
		)

	diagnosis_list = ['SR', 'AFIB'] # ['NORM', 'MI', 'STTC', 'HYP', 'CD']

	if train_model_per_class:

		for tgt in range(ptb_classes):

			confusion_matrix = torch.zeros((n_classes, 4)).to(device)

			for j, data in enumerate(train_loader):

				samples, labels = data[0].to(device), \
					data[1].to(device)

				with torch.no_grad():

					if mix_dimensions and not one_dimensional:

						pred = encoder(samples)
						pred, _ = seg_mask_to_signal(pred)
						if pred.shape[0] == 187:
							pred = torch.zeros(samples.shape[0], 1)
						else:
							pred = model(pred)

					else:

						pred = model(samples)

						if training_task == 'rule_based':

							y = pred.cpu().numpy()
							pred = np.zeros((y.shape[0], 1))
							for pr in range(y.shape[0]):
								pred[pr,] = predict_ecg(y[pr,])[tgt]

				labels = labels[:, nc : nc + 1]
						
				for h in range(pred.shape[0]):
					for i in range(1):
						if pred[h, i] > 0.5 and labels[h, i] > 0.5:
							confusion_matrix[i, 0] += 1
						elif pred[h, i] < 0.5 and labels[h, i] < 0.5:
							confusion_matrix[i, 3] += 1
						elif pred[h, i] > 0.5 and labels[h, i] < 0.5:
							confusion_matrix[i, 1] += 1
						elif pred[h, i] < 0.5 and labels[h, i] > 0.5:
							confusion_matrix[i, 2] += 1


					print('Predicted batch', j, end='\r')

			if test:
				print('\nPredicted test set, saving results...')
			else:
				print('\nPredicted validation set, saving results...')

			
			diag_list = diagnosis_list[tgt]

			csv = ""
			for d in diag_list:
				csv += "," + d
			
			metric_list = ['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'F1']
			metric_matrix = torch.zeros((n_classes, 4)).to(device)
			metric_matrix[:, 0] = confusion_matrix[:, 0] / (confusion_matrix[:, 0] + confusion_matrix[:, 2] + 1e-8)
			metric_matrix[:, 1] = confusion_matrix[:, 3] / (confusion_matrix[:, 1] + confusion_matrix[:, 3] + 1e-8)
			metric_matrix[:, 2] = confusion_matrix[:, 0] / (confusion_matrix[:, 0] + confusion_matrix[:, 1] + 1e-8)
			metric_matrix[:, 3] = (metric_matrix[:, 0] * metric_matrix[:, 2] * 2) / (metric_matrix[:, 0] + metric_matrix[:, 2])

			for i in range(4):
				csv += "\n" + metric_list[i]
				for h in range(n_classes):
					csv +="," + str(confusion_matrix[h, i].item())
			for i in range(4):
				csv += "\n" + metric_list[i]
				for h in range(n_classes):
					csv +="," + str(metric_list[h, i].item())

			if test:
				with open(config.results_folder + splitter + \
						"test_results_" + file_suffix + '_' + str(tgt) + ".csv", "w") as f:
					f.write(csv)
			else:
				with open(config.results_folder + splitter + \
						"validation_results_" + file_suffix + '_' + str(tgt) + ".csv", "w") as f:
					f.write(csv)

	else:

		confusion_matrix = torch.zeros((n_classes, 4)).to(device)

		# temporary
		counter = 0

		for j, data in enumerate(train_loader):

			samples, labels = data[0].to(device), \
				data[1].to(device)

			with torch.no_grad():

				if mix_dimensions and not one_dimensional:

					pred = encoder(samples)
					pred, _ = seg_mask_to_signal(pred)
					if pred.shape[0] == 187:
						pred = torch.zeros(samples.shape[0], ptb_classes)
					else:
						pred = model(pred)

				else:

					if training_task == 'rule_based':

						pred = model(samples).cpu().numpy()

						if not os.path.isdir(config.folder + '/masks') and not test:
							os.mkdir(config.folder + '/masks')
						np.save(config.folder + '/masks/' + str(counter).zfill(4) + '_' + str(labels[0,0].item()) + '_.npy', pred)
						counter += 1

						y = np.zeros((pred.shape[0], 2))
						for pr in range(y.shape[0]):
							y[pr,] = predict_ecg(pred[pr,], labels[pr,].cpu().numpy(), test)
						pred = y

					else:

						pred = model(samples)						
					
			for h in range(pred.shape[0]):
				for i in range(ptb_classes):
					if pred[h, i] > 0.5 and labels[h, i] > 0.5:
						confusion_matrix[i, 0] += 1
					elif pred[h, i] < 0.5 and labels[h, i] < 0.5:
						confusion_matrix[i, 3] += 1
					elif pred[h, i] > 0.5 and labels[h, i] < 0.5:
						confusion_matrix[i, 1] += 1
					elif pred[h, i] < 0.5 and labels[h, i] > 0.5:
						confusion_matrix[i, 2] += 1

				print('Predicted batch', j, end='\r')

		if test:
			print('\nPredicted test set, saving results...')
		else:
			print('\nPredicted validation set, saving results...')

		csv = ""
		for d in diagnosis_list:
			csv += "," + d
		
		metric_list = ['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'F1']
		metric_matrix = torch.zeros((n_classes, 4)).to(device)
		metric_matrix[:, 0] = confusion_matrix[:, 0] / (confusion_matrix[:, 0] + confusion_matrix[:, 2] + 1e-8)
		metric_matrix[:, 1] = confusion_matrix[:, 3] / (confusion_matrix[:, 1] + confusion_matrix[:, 3] + 1e-8)
		metric_matrix[:, 2] = confusion_matrix[:, 0] / (confusion_matrix[:, 0] + confusion_matrix[:, 1] + 1e-8)
		metric_matrix[:, 3] = (metric_matrix[:, 0] * metric_matrix[:, 2] * 2) / (metric_matrix[:, 0] + metric_matrix[:, 2])

		for i in range(4):
			csv += "\n" + metric_list[i]
			for h in range(n_classes):
				csv +="," + str(confusion_matrix[h, i].item())
		for i in range(4):
			csv += "\n" + metric_list[i + 4]
			for h in range(n_classes):
				csv +="," + str(metric_matrix[h, i].item())

		if test:
			with open(config.results_folder + splitter + \
					"test_results_" + file_suffix + ".csv", "w") as f:
				f.write(csv)
		else:
			with open(config.results_folder + splitter + \
					"validation_results_" + file_suffix + ".csv", "w") as f:
				f.write(csv)

		if training_task == 'rule_based' and not test:
			brute_force_search()


##############################################################
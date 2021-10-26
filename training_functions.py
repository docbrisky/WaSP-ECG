import random
import cv2
from config import config
import glob
import numpy as np
from scipy.signal import find_peaks
from image_augmentor import Image_Augmentor
from ecg_generator import training_generator, training_generator_signal_only
from helper_functions import indices_to_one_hot, n_classes
import torch
import torch.nn as nn
from torchvision import  models
from helper_functions import n_classes, points, waves
import pickle
from ptb_loader import Physionet_Helper
import time
from original_image_augmentor import augment_image

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

class Classifier(torch.nn.Module):

	def __init__(self, unet, output_classes=7, penultimate=False):

		super(Classifier, self).__init__()
		self.avepool = torch.nn.AdaptiveMaxPool2d(1)
		self.flatten = torch.nn.Flatten()
		self.unet = unet
		self.penultimate = penultimate

		with torch.no_grad():
			x = torch.zeros((1, 3, config.dim, config.dim))
			x = unet.encoder(x)
			x = self.avepool(x[-1])
			x = self.flatten(x)
			self.shape = x.shape[-1]

		self.fc1 = torch.nn.Linear(self.shape, 512)
		self.fc2 = torch.nn.Linear(512, output_classes)
		self.relu = torch.nn.ReLU()

	def forward(self, x):

		if self.penultimate:

			x = self.unet.encoder(x)
			x = self.avepool(x[-1])
			x = self.flatten(x)

			return x

		else:

			x = self.unet.encoder(x)
			x1 = self.unet.decoder(*x)
			x1 = self.unet.segmentation_head(x1)
			x = self.avepool(x[-1])
			x = self.flatten(x)
			x = self.fc1(x)
			x = self.relu(x)
			x = self.fc2(x)
			
			return x1, torch.nn.functional.sigmoid(x)

	def encode(self, x):

		return self.unet.encoder(x)

	def get_shape(self):

		return self.shape

class Classifier_ptb(torch.nn.Module):

	def __init__(self, base_model, output_classes=5, full_retrain=True):

		super(Classifier_ptb, self).__init__()
		self.base_model = base_model
		self.fc1 = torch.nn.Linear(base_model.get_shape(), 1024)
		self.fc2 = torch.nn.Linear(1024, 512)
		self.fc3 = torch.nn.Linear(512, 256)
		self.fc4 = torch.nn.Linear(256, output_classes)
		self.relu = torch.nn.ReLU()
		self.full_retrain = full_retrain

	def forward(self, x):

		if self.full_retrain:
			x = self.base_model(x)

		else:
			with torch.no_grad():
				x = self.base_model(x)
				
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.relu(x)
		x = self.fc4(x)
		
		return torch.nn.functional.sigmoid(x)

	def encoder(self, x):

		return self.base_model.unet.encoder(x)

class DataGenerator():

	def __init__(
			self,
			batch_size=1,
			folder='', 
			bg_image_list=None,
			dim=1024, 
			n_channels=3, 
			counter=0, 
			ECG_suffix='ECG', 
			label_suffix='label', 
			noise_multiplier=3., 
			impedance=2., 
			augment=False, 
			rhythms=['sr', 'af'], 
			rates=[40, 150], 
			p_multiplier=0.3, 
			t_multiplier=2., 
			presets=['Normal', 'LAHB', 'LPHB', 'high_take_off', 'LBBB', 'ant_STEMI'], 
			save_data=False,
			splitter='/',
			waves_only=False,
			broad_narrow_balance=False,
			segment_leads=False,
			one_dimensional=False
			):

		if save_data:
			self.dim = config.save_dim
		else:
			self.dim = dim
		if folder != '':
			folder = folder + splitter
		self.folder = folder
		self.bg_image_list = bg_image_list
		self.n_channels = n_channels
		self.counter = counter
		self.ECG_suffix = ECG_suffix
		self.label_suffix = label_suffix
		self.noise_multiplier = noise_multiplier
		self.impedance = impedance
		self.augment = augment
		self.rhythms = rhythms
		self.rates = rates
		self.p_multiplier = p_multiplier
		self.t_multiplier = t_multiplier
		self.presets = presets
		self.save_data = save_data
		self.batch_size = batch_size
		self.waves_only = waves_only
		self.img_aug = Image_Augmentor(
			dim = dim, 
			n_classes = n_classes, 
			augment = augment,
			waves_only = waves_only
			)
		self.broad_narrow_balance = broad_narrow_balance
		self.segment_leads = segment_leads
		self.one_dimensional = one_dimensional

	def __getitem__(self, index):

		random.seed(time.time())

		rhythm_int = random.randint(0, len(self.rhythms) - 1)
		rhythm = [self.rhythms[rhythm_int]]
		if index % 2 == 0 and self.broad_narrow_balance:
			preset = 'LBBB'
			preset_int = 4
		else:
			preset_int = random.randint(0, len(self.presets) - 1)
			preset = [self.presets[preset_int]]

		if self.one_dimensional:

			X, y, label_vector = training_generator_signal_only(
				universal_noise_multiplier = self.noise_multiplier, 
				impedance = self.impedance, 
				rhythms = rhythm, 
				rates = self.rates, 
				p_multiplier = self.p_multiplier, 
				t_multiplier = self.t_multiplier, 
				presets = [preset],
				include_points = not self.waves_only,
				segment_leads = self.segment_leads,
				new_random_seed = True
				)

			X = X.reshape(-1)
			y = y.reshape(-1)

			y = indices_to_one_hot(
				y, 
				n_classes,
				waves_only = self.waves_only,
				one_dimensional = True
				)

		else:

			X, y, _ = training_generator(
				universal_noise_multiplier = self.noise_multiplier, 
				impedance = self.impedance, 
				rhythms = rhythm, 
				rates = self.rates, 
				p_multiplier = self.p_multiplier, 
				t_multiplier = self.t_multiplier, 
				presets = [preset],
				include_points = not self.waves_only,
				segment_leads = self.segment_leads,
				new_random_seed = True
				)

			X, y = self.img_aug.transform_image_mask(X, y)

		y = y.astype(np.uint8)
		y = np.packbits(y)

		z = np.zeros(7)
		z[preset_int] = 1
		z[-1] = rhythm_int

		if self.save_data:
			np.save(self.folder + str(index).zfill(4) + '_ECG.npy', X)
			np.save(self.folder + str(index).zfill(4) + '_label.npy', y)
			np.save(self.folder + str(index).zfill(4) + '_onehot.npy', z)

		random.seed(config.random_seed)

		return X, y

	def __len__(self):

		return self.buffer_size

class DS_Loader(torch.utils.data.Dataset):

	def __init__(
			self, 
			file_list,
			splitter='/',
			point_loader=False,
			one_dimensional = False,
			n_presets = 7,
			label_vec_len = 62,
			ptb = False,
			ptb_val = False,
			ptb_test = False
			):

		self.file_list = file_list
		self.ECG_suffix = config.ECG_suffix
		self.label_suffix = config.label_suffix
		self.folder = config.folder
		self.dim = config.dim
		self.splitter = splitter
		self.point_loader = point_loader

		if config.deepspeed:

			self.half_precision = config.use_half

		else:
			self.half_precision = False

		self.img_aug = Image_Augmentor(
			dim = self.dim, 
			n_classes = n_classes, 
			augment = config.augment,
			waves_only = config.waves_only
			)
		self.one_dimensional = one_dimensional
		self.n_presets = n_presets
		self.ptb = ptb

		if ptb:
			if ptb_val:
				self.folder = config.ptb_val_folder
			elif ptb_test:
				self.folder = config.ptb_test_folder
			else:
				self.folder = config.ptb_folder

		print('Data loader pointed at:', self.folder)

	def __getitem__(self, index):

		X = np.load(self.folder + self.splitter + self.file_list[index] \
				+ '_' + self.ECG_suffix + '.npy')
		y = np.load(self.folder + self.splitter + self.file_list[index] \
				+ '_' + self.label_suffix + '.npy')

		if not self.ptb:

			z = np.load(self.folder + self.splitter + self.file_list[index] \
					+ '_onehot.npy')

		if self.one_dimensional:

			# segmentation masks contain an extra class to mark the boundary between beats,
			# so saved masks have n_classes + 1 classes
			y = np.unpackbits(y).reshape(X.shape[0], n_classes + 1)
			X = torch.from_numpy(np.expand_dims(X, axis=0))
			y = torch.from_numpy(np.moveaxis(y, -1, 0))

		else:

			if self.ptb:

				X = self.img_aug.transform_image(X)

			else:

				# segmentation masks contain an extra class to mark the boundary between beats:
				y = np.unpackbits(y).reshape(X.shape[0], X.shape[1], n_classes + 1)

				X, y = self.img_aug.transform_image_mask(X, y)
				# # to run enhanced augmentation, comment out the line above and uncomment the line below:
				# X, y = augment_image(X, y, target_size = self.dim)

			y = torch.from_numpy(np.moveaxis(y, -1, 0)).type(torch.float)
			X = torch.from_numpy(np.moveaxis(X, -1, 0)).type(torch.float) / 255

		if not self.ptb:

			z = torch.from_numpy(z)
		
		if self.half_precision:
			X = X.type(torch.half)
			y = y.type(torch.half)

			if not self.ptb:

				z = z.type(torch.half)

		else:
			X = X.type(torch.float)
			y = y.type(torch.float)

			if not self.ptb:

				z = z.type(torch.float)

		if config.deepspeed:

			if not self.ptb:

				return X.contiguous(), y.contiguous(), z.contiguous()

			else:

				return X.contiguous(), y.contiguous()

		else:

			if not self.ptb:

				return X, y, z

			else:

				return X, y

	def __len__(self):

		return len(self.file_list)

class Physionet_Loader(torch.utils.data.Dataset):

	def __init__(self, plot=False, val=False, test=False, balance=False):

		self.ph = Physionet_Helper()
		self.train_list = self.ph.get_train_list()
		if val:
			self.train_list = self.ph.get_val_list()
		elif test:
			self.train_list = self.ph.get_test_list()
		elif balance:
			self.train_list = self.ph.balance_samples(self.train_list)
		self.plot = plot
		if self.plot:
			self.img_aug = Image_Augmentor(
				dim = config.dim, 
				n_classes = n_classes, 
				augment = config.augment,
				waves_only = config.waves_only
				)

	def __getitem__(self, index):

		x, y = self.ph.load_sample_ecg(self.train_list[index], plot=self.plot)

		if not self.plot:

			x = np.expand_dims(x.reshape(-1), axis=0)

		else:

			x, y = self.img_aug.transform_image_mask(x, y)

			x = x / 255

		x = torch.from_numpy(x).type(torch.float)

		y = torch.from_numpy(y).type(torch.float)

		return x, y

	def __len__(self):

		return len(self.train_list)

def balance_samples(sample_numbers, n_classes, target_class, folder, val=False):

	samples_array = np.zeros((len(sample_numbers), n_classes))
	for s in range(len(sample_numbers)):
		samples_array[s,] = np.load(folder + '/' + sample_numbers[s] + '_label.npy')
	samples_array = samples_array[:, target_class]
	pos_list = []
	neg_list = []

	for s in range(samples_array.shape[0]):
		if samples_array[s] == 0:
			neg_list.append(sample_numbers[s])
		else:
			pos_list.append(sample_numbers[s])

	random.shuffle(pos_list)
	random.shuffle(neg_list)

	if not val:

		if len(neg_list) > len(pos_list):
			while len(neg_list) > len(pos_list):
				pos_list += pos_list
			pos_list = pos_list[:len(neg_list)]
		else:
			while len(pos_list) > len(neg_list):
				neg_list += neg_list
			neg_list = neg_list[:len(pos_list)]
			
		sample_numbers = neg_list + pos_list

	random.shuffle(sample_numbers)

	return sample_numbers

def f1_torch(y_true, y_pred):

	f1_list = []
	f1_classes = n_classes
	for i in range(f1_classes):
		yt = y_true[:, i, ]
		yp = y_pred[:, i, ]
		prec = precision_torch(yt, yp)
		rec = recall_torch(yt, yp)
		f1_list.append(2 * ((prec * rec) / (prec + rec + torch.tensor([1e-8], device=y_true.device))))

	return sum(f1_list) / len(f1_list)

def multiprocess_loader(arg_list, dim=None):

	if dim == None:
		dim = config.dim

	X = np.load(arg_list[0])
	y = np.load(arg_list[1])
	y = np.unpackbits(y).reshape(X.shape[0], X.shape[1], n_classes)

	X, y = arg_list[2].transform_image_mask(X, y)

	return X, y

def multiprocess_generator(arg_list):

	return arg_list[0].__getitem__(arg_list[1])

def multitask_loop(
	train_loader,
	criterion,
	seg_criterion,
	optimizer,
	optimizer_enc,
	one_dimensional=False,
	buffer_size=64,
	total_samples=0,
	use_deepspeed=False,
	model=None,
	encoder=None,
	model_engine=None,
	f1_list=[],
	loss_list=[],
	device=None
	):

	# initialise running metrics:
	running_loss = 0
	running_seg_loss = 0
	f1_score = 0

	for h, data in enumerate(train_loader):

		if use_deepspeed:

			samples, seg_mask, labels = \
				data[0].to(model_engine.local_rank), \
				data[1].to(model_engine.local_rank), \
				data[2].to(model_engine.local_rank)

		else:

			samples, seg_mask, labels = \
				data[0].to(device), \
				data[1].to(device), \
				data[2].to(device)

		seg_mask = seg_mask[:, :-1, ]

		if use_deepspeed:

			# retain_graph not implemented for DS at time of writing
			# calculate loss & gradients for segmentation first:
			seg_pred, _ = model_engine(samples)
			f1_score_temp = f1_torch(seg_pred, seg_mask) / buffer_size
			f1_score += f1_score_temp.item()
			loss = seg_criterion(seg_pred, seg_mask) / buffer_size
			running_seg_loss += loss.item()
			model_engine.backward(loss)
			del seg_pred

			# then repeat for classification:
			_, pred = model_engine(samples)
			loss = criterion(pred, labels) / buffer_size
			running_loss += loss.item()
			model_engine.backward(loss)

		else:

			optimizer.zero_grad()
			
			if one_dimensional:

				optimizer_enc.zero_grad()
				seg_pred = encoder(samples)
				pred = encoder.encode(samples)
				pred = model(pred)

			else:

				seg_pred, pred = model(samples)

			f1_score_temp = f1_torch(seg_pred, seg_mask) / buffer_size
			f1_score += f1_score_temp.item()

			seg_loss = seg_criterion(seg_pred, seg_mask) / buffer_size
			running_seg_loss += seg_loss.item()

			loss = criterion(pred, labels) / buffer_size
			running_loss += loss.item()

			loss.backward(retain_graph=True)
			seg_loss.backward()

		if h % buffer_size == 0 and h != 0:

			if use_deepspeed:

				model_engine.step()
				print_txt = 'DS '

			else:

				optimizer.step()
				if one_dimensional:
					optimizer_enc.step()
				print_txt = 'Torch '

			running_loss = running_loss

			if one_dimensional:
				print_txt += '1D: '
			else:
				print_txt += '2D: '
			print_txt += 'Loss: ' + str(running_loss)
			print_txt += ' Seg loss: ' + str(running_seg_loss)
			print_txt += ' F1 score: ' + str(f1_score)
			print_txt += ' for batch ' + str(h) + ' of ' + str(total_samples)
			print(print_txt)

			f1_list.append(f1_score)
			loss_list.append(running_loss)

			f1_score = 0
			running_loss = 0
			running_seg_loss = 0

	if use_deepspeed:

		return [model_engine], f1_list, loss_list

	elif one_dimensional:

		return [encoder, model], f1_list, loss_list

	else:

		return [model], f1_list, loss_list

def precision_torch(y_true, y_pred):

	true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
	predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + torch.tensor([1e-8], device=y_true.device))

	return precision

def recall_torch(y_true, y_pred):

	true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
	possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + torch.tensor([1e-8], device=y_true.device))

	return recall

def seg_mask_to_signal(mask, width=100, length=3000, n_lines=3):

	device = mask.device

	mask = mask.cpu()

	signal = torch.zeros((mask.shape[0], 1, length * n_lines))

	for i in range(mask.shape[0]):

		y_np = torch.argmax(mask[i,], dim=0).numpy()
		lines = np.sum(y_np == 3, axis = 1)
		peaks, _ = find_peaks(lines, distance=width)

		signal_list = []

		if peaks.shape[0] == 4:

			y = torch.argmax(mask[i,], dim=0)
			y = torch.moveaxis(y, 0, 1)

			for p in range(n_lines):
				
				x_indices, y_indices = torch.nonzero(y[:, peaks[p] - 90 : peaks[p] + 90], as_tuple=True)

				y_point_list = []
				x_point_list = []
				ave_list = []
				for idx in range(x_indices.shape[0]):
					if x_indices[idx].item() not in x_point_list and idx != 0:
						y_point_list.append(sum(ave_list) / len(ave_list))
						ave_list = [y_indices[idx].item()]
						x_point_list = [x_indices[idx]]
					else:
						ave_list.append(y_indices[idx].item())
						x_point_list.append(x_indices[idx])

				signal_list += y_point_list

		else:

			return torch.zeros(187), peaks

	signal_list = torch.FloatTensor(signal_list)

	gap = (length * n_lines) - signal_list.shape[0]

	signal[i, :, : - gap] = signal_list

	signal = (signal.to(device) - 90) * 2e-5

	return signal, None

def single_task_loop(
	train_loader,
	criterion,
	optimizer,
	buffer_size=64,
	total_samples=0,
	use_deepspeed=False,
	model=None,
	model_engine=None,
	encoder=None,
	f1_list=[],
	loss_list=[],
	training_task='segmentation',
	target_class=-1,
	device=None,
	one_dimensional = False,
	mix_dimensions = False
	):

	if one_dimensional:
		dim_text = '1D'
		use_deepspeed = False
	else:
		dim_text = '2D'

	if training_task == 'segmentation':
		segmentation = True
	else:
		segmentation = False

	# initialise running metrics:
	f1_score = 0
	running_loss = 0

	for h, data in enumerate(train_loader):
		# a loop breaker given that image-signal transposition is note completely robust:
		passer = False

		# push data to GPU:
		if use_deepspeed:
			samples, labels = data[0].to(model_engine.local_rank), \
				data[1].to(model_engine.local_rank)
		else:
			samples, labels = data[0].to(device), \
				data[1].to(device)
		# predict outputs:
		if not segmentation and mix_dimensions:
			with torch.no_grad():
				pred = encoder(samples)
		else:
			if use_deepspeed:
				pred = model_engine(samples)
			else:
				pred = model(samples)

		if segmentation:
			# remove the final class (exists for a different experiment):
			labels = labels[:, :-1,]
			# calculate F1 score:
			f1_score_temp = f1_torch(pred, labels) / buffer_size
			f1_score += f1_score_temp.item()
		else:
			# it is possible to fine-tune a new model for each target class:
			if target_class != -1:
				labels = labels[:, target_class : target_class + 1]
			# different classification approach for image inputs:
			if mix_dimensions:
				# the 2D classification extracts a signal vector from the segmentation mask:
				pred, peaks = seg_mask_to_signal(pred.detach())
				# 187 is just a random number that is unlikely to equal the batch size:
				if pred.shape[0] == 187:
					passer = True
					pass
				elif use_deepspeed:
					pred = model_engine(pred)
				else:
					pred = model(pred)

		if not passer:
			# calculate loss:
			loss = criterion(pred, labels) / buffer_size
			running_loss += loss.item()

			if use_deepspeed:
				# calculate gradients:
				model_engine.backward(loss)
				# run backprop with accumulated gradients if buffer limit reached:
				if h % buffer_size == 0 and h != 0:
					if model_engine.local_rank == 0:
						if segmentation:
							f1_list.append(f1_score)
							print('DS segmentation ' + dim_text + \
								': Loss:', running_loss, 'F1:', f1_score, 'for batch', h, 'of', total_samples)
						else:
							print('DS classification ' + dim_text + \
								': Loss:', running_loss, 'for batch', h, 'of', total_samples)
					model_engine.step()
					loss_list.append(running_loss)
					running_loss = 0
					f1_score = 0
			else:
				# calculate gradients:
				loss.backward()
				# run backprop with accumulated gradients if buffer limit reached:
				if h % buffer_size == 0 and h != 0:
					optimizer.step()
					optimizer.zero_grad()
					loss_list.append(running_loss)
					if segmentation:
						print_txt = 'Torch segmentation ' + dim_text + ': '
					else:
						print_txt = 'Torch classification ' + dim_text + ': '
					print_txt += 'Loss: ' + str(running_loss)
					if segmentation:
						f1_list.append(f1_score)
						print_txt += ' F1:' + str(f1_score)
					print_txt += ' for batch ' + str(h) + ' of ' + str(total_samples)
					print(print_txt)
					running_loss = 0
					f1_score = 0

	if use_deepspeed:

		return [model_engine], f1_list, loss_list

	else:

		return [model], f1_list, loss_list

def shorten_signal(X, y):

	new_X = torch.empty((X.shape[0], X.shape[1], 36000), device=X.device)
	new_y = torch.empty((y.shape[0], y.shape[1], 36000), device=y.device)

	for i in range(12):

		new_X[:, :, i * 3000 : (i + 1) * 3000] = X[:, :, i * 10000 : (i * 10000) + 3000]
		new_y[:, :, i * 3000 : (i + 1) * 3000] = y[:, :, i * 10000 : (i * 10000) + 3000]


	X = new_X[:, :, ::4]
	y = new_y[:, :, ::4]

	return X, y
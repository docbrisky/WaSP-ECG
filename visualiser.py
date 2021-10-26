from config import config
import numpy as np
from helper_functions import class_colors, find_font, n_classes, segmentation_key
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import pickle
from image_augmentor import Image_Augmentor
import torch
import segmentation_models_pytorch as smp
from unet_1d import UNET_1D, Classification_Head, PTB_Model
from ecg_generator import plot_12_lead_ecg, training_generator, training_generator_signal_only
import glob
import random
from datetime import datetime
from training_functions import Classifier, Classifier_ptb, seg_mask_to_signal

def add_key(image):

	paper_height = 7800
	paper_width = 11200

	image_width, image_height = image.size

	height_ratio = image_height / paper_height
	width_ratio = image_width / paper_width

	font = find_font()

	draw = ImageDraw.Draw(image)
	rect_height_start = 50 * height_ratio
	rect_width_start = 9000 * width_ratio
	rect_diam = 100 * height_ratio * 1.5
	pil_font = ImageFont.truetype(font,  
		size = int(48 * height_ratio * 2), 
		encoding = "unic")
	for classes in range(1, n_classes, 1):
		rect_height_start += int(rect_diam * 1.5)
		draw.rectangle([rect_width_start, rect_height_start,
					rect_width_start + (rect_diam * 10), rect_height_start + rect_diam], 
				fill = "rgb(255,255,255)")
		draw.rectangle([rect_width_start, rect_height_start,
					rect_width_start + rect_diam, rect_height_start + rect_diam], 
				fill = "rgb(" + str(class_colors[classes][0]) + "," \
					+ str(class_colors[classes][1]) + "," \
					+ str(class_colors[classes][2]) + ")")
		draw.text((rect_width_start + rect_diam + (30 * width_ratio), rect_height_start + (10 * height_ratio)), 
			segmentation_key[classes], 
			font = pil_font, 
			fill = "#0000FF")

	return image

def visualise(model_path, out_path, predict=True, create_new=False, decoder_path=None, ptb_signal=None, file=''):

	# set CUDA device:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	mask_switch = False

	two_d = False

	if '2d' in model_path:
		two_d = True

	folder = config.ptb_folder
	dim = config.dim

	print('Folder:', folder)

	im_aug = Image_Augmentor(n_classes = n_classes, waves_only = config.waves_only, dim = dim)

	use_torch = True
	use_deepspeed = config.deepspeed

	orig_h = dim
	orig_w = dim

	if create_new:

		if two_d:

			x, mask, _ = training_generator(new_random_seed=True, include_points=not config.waves_only)
			im, mask = im_aug.transform_image_mask(x, mask)

		else:

			im, mask = training_generator_signal_only(new_random_seed=True, include_points=not config.waves_only)

		mask = np.argmax(mask, axis=-1)

	elif not two_d and predict:

		x = ptb_signal
		x = x.reshape(-1)

	else:

		if file == '':
			print('No file set, picking from folder:', folder)
			random.seed(datetime.now())
			files = glob.glob(folder + '/*_ECG.npy')
			ecg_file = random.choice(files)
		else:
			ecg_file = file
			files = [file]

		if len(files) == 0:
			print('Error, no ECG files found!')
			quit()
		else:
			print('Loading ECG at ' + ecg_file)
			if '.npy' in ecg_file:
				x = np.load(ecg_file)
			elif '.jpg' in ecg_file:
				x = cv2.imread(ecg_file)
				x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
			orig_h = x.shape[0]
			orig_w = x.shape[1]

		label_file = folder + '/' + ecg_file.split('/')[-1].split('_')[0] + '_label.npy'

		if two_d:

			if not predict:
		
				if os.path.isfile(label_file):
					mask = np.unpackbits(np.load(label_file)).reshape((x.shape[0], x.shape[1], n_classes + 1))
					mask = np.argmax(mask, axis=-1)
					mask_switch = True
					if mask.shape[0] != dim or mask.shape[1] != dim:
						print('Resizing mask...')
						mask = cv2.resize(mask, (dim, dim), interpolation = cv2.INTER_NEAREST)[:, :, 0]

			x = np.asarray(x)
			if x.shape[0] != dim or x.shape[1] != dim:
				print('Resizing ECG...')
				x = cv2.resize(x, (dim, dim), interpolation = cv2.INTER_AREA)[:, :, :3]
				
			im = x
			im = np.uint8(im)

		elif not predict:

			if os.path.isfile(label_file):
				mask = np.unpackbits(np.load(label_file)).reshape((x.shape[0], n_classes + 1))
				mask_switch = True
				mask = mask.reshape(-1)
				mask = mask.reshape((12, 10000))
				x = x.reshape(-1)
				x = x.reshape((12, 10000))
				x, mask = plot_12_lead_ecg(x, mask, '', print_meta_data=False)
				im, mask = im_aug.transform_image_mask(x, mask)

	# predict segmentation mask
	if predict:

		if two_d:

			model = smp.Unet(
					encoder_name = config.model,
					encoder_weights = config.encoder_weights,
					in_channels = 3,
					classes = n_classes,
					decoder_attention_type = "scse",
					activation = "softmax"
					)

			if 'classification' in model_path:

				encoder = model
				encoder.load_state_dict(torch.load(config.model_folder + '/unet_2d.p'))
				model = Classifier(encoder, output_classes = 7, penultimate = True)
				model = Classifier_ptb(model, output_classes = 2)

			if os.path.isfile(model_path):
				model.load_state_dict(torch.load(model_path))
			model.eval()
			model = model.to(device)

			model_input = np.moveaxis(np.expand_dims(im, axis=0) / 255, -1, 1)
			model_input = torch.from_numpy(model_input).type(torch.float)
			model_input = model_input.to(device)

			with torch.no_grad():
				if 'classification' in model_path:
					pr = model.encoder(model_input)
					pr = encoder.decoder(*pr)
					pr = encoder.segmentation_head(pr).cpu().numpy()[0,]
				else:
					pr = model(model_input).cpu().numpy()[0,]
			pr = np.moveaxis(pr, 0, -1)
			pr = np.argmax(pr, axis=-1)

		else:

			model = UNET_1D(
				n_classes
			) 

			if 'classification' in model_path:

				encoder = model
				encoder.load_state_dict(torch.load(config.model_folder + '/unet_1d.p'))
				head = Classification_Head(encoder = encoder, n_classes = 7)
				model = PTB_Model(encoder, head, n_classes = 2)

			model.load_state_dict(torch.load(model_path))
			model.eval()
			model = model.to(device)

			model_input = torch.from_numpy(np.expand_dims(np.expand_dims(x, axis=0), axis=0)).type(torch.float)
			model_input = model_input.to(device)
			with torch.no_grad():
				if 'classification' in model_path:
					y = model.encode(model_input)
					y = encoder.decoder(y).cpu().numpy()[0,]
				else:
					y = model(model_input).cpu().numpy()[0,]
				
			y = np.argmax(y, axis=0)
			y = y.reshape(-1)
			y = y.reshape((12, 10000))
			x = x.reshape(-1)
			x = x.reshape((12, 10000))
			im, pr = plot_12_lead_ecg(x, y, '', print_meta_data=False, save_image=False)
			im, pr = im_aug.transform_image_mask(im, pr)		
			pr = np.argmax(pr, axis=-1)

		seg_img = np.zeros((dim, dim, 3))
		for c in range(1, n_classes, 1):
			seg_img[:, :, 0] += (pr[:, :] == c) * (class_colors[c][0])
			seg_img[:, :, 1] += (pr[:, :] == c) * (class_colors[c][1])
			seg_img[:, :, 2] += (pr[:, :] == c) * (class_colors[c][2])

		edited_image = Image.fromarray(np.uint8(cv2.addWeighted(im, 
							0.5, np.uint8(seg_img), 0.5, 0)))
		edited_image = add_key(edited_image)
		edited_image = edited_image.resize((orig_w, orig_h))
		edited_image.save(out_path, 'JPEG')

	elif mask_switch:
	
		seg_img = np.zeros((dim, dim, 3))

		for c in range(1,n_classes,1):

			seg_img[:, :, 0] += (mask[:, :] == c)*(class_colors[c][0])
			seg_img[:, :, 1] += (mask[:, :] == c)*(class_colors[c][1])
			seg_img[:, :, 2] += (mask[:, :] == c)*(class_colors[c][2])

		edited_image = Image.fromarray(np.uint8(cv2.addWeighted(im, 
							0.5, np.uint8(seg_img), 0.5, 0)))
		edited_image = add_key(edited_image)
		edited_image.save(out_path, 'JPEG')

	print('Image written to ' + out_path)

if __name__  ==  '__main__':
	
	# from ptb_loader import Physionet_Helper
	# ph = Physionet_Helper()
	# train_list = ph.get_train_list()
	# signal, _ = ph.load_sample_ecg(train_list[0], plot=False)
	# # visualise('models/unet_1d.p', 'sample_1d.jpg', ptb_signal=signal)
	# visualise('models/unet_1d_SR_AF_classification.p', 'sample_1d.jpg', ptb_signal=signal)

	# visualise('2d', 'sample_2d_naive.jpg', create_new=True)
	# visualise('models/unet_2d_SR_AF_classification.p', 'sample_2d_key.jpg')

	files = glob.glob('/media/rbrisk/storage01/masks/*1.0_test.npy')
	visualise('models/unet_2d_SR_AF_classification.p', 'sample_2d_key.jpg', file = '/media/rbrisk/storage01/ptb/5304_ECG.npy')

	# files = glob.glob('/media/rbrisk/storage01/masks/*1.0_test.npy')
	# visualise('models/unet_2d_augmented.p', 'sample_2d_scan_002.jpg', file = 'real_ECGs/ECG_scan_002.jpg')

	# files = glob.glob('/media/rbrisk/storage01/masks/*1.0_test.npy')
	# visualise('models/unet_2d_augmented.p', 'sample_2d_scan_003.jpg', file = 'real_ECGs/ECG_scan_003.jpg')
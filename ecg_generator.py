import numpy as np
import random
from rhythm_generator import nsr,  af
import time
import os
import argparse
import random
from PIL import Image,  ImageDraw,  ImageFont
import cv2
import string
from pathlib import Path
import multiprocessing
from helper_functions import evenly_spaced_y, \
	ecg_presets, \
	line_drawer, \
	create_thickened_label, \
	scale_array, \
	create_paper, \
	find_font, \
	class_colors, \
	segmentation_key
from config import config
random.seed(config.random_seed)

def plot_12_lead_ecg(
		y, 
		y_label, 
		meta_data, 
		filename='test_ecg', 
		image_type='PNG', 
		label_name='test_label', 
		print_meta_data=True, 
		rhythm_strip=1, 
		random_text=False, 
		random_layout=False, 
		save_image=True, 
		thicken_trace=True,
		include_key=False,
		print_merged_image_only=True,
		lead_segmentation=False,
		save_dim=0
		):

	debugging = False

	if debugging:
		print('Y shape', y.shape)
		print('Y label shape', y_label.shape)
		print('Min value', np.amin(y_label), 'Max value', np.amax(y_label))

	if (y.shape[0] !=  12 or y.shape[1] !=  10000):
		print('Error, this program only prints 12 lead ECG signals recorded at 1000Hz')
		print('Your signal contains ' + str(y.shape[0]) + ' leads at ' + str(y.shape[1] // 10) + 'Hz')

	else:
		if lead_segmentation:
			for i in range(y_label.shape[0]):
				y_label[i, :] = i + 1

		rhythm_strip = random.randint(0, y.shape[0] - 1)

		if debugging:
			print('Loading paper...')
			start = time.perf_counter()
		if os.path.isfile('paper.npy') == False:
			create_paper()
		paper = np.load('paper.npy')
		paper_height = 7800
		paper_width = 11200
		label = np.zeros((paper_height, paper_width))

		if debugging:
			print('Loading paper took', time.perf_counter() - start, 'seconds')
			start = time.perf_counter()

		# draw annotations
		if debugging:
			print('Drawing annotations...')
		if random_layout:
			top_offset = random.randint(1000, 2600)
			side_offset = random.randint(0, 1000)
			lead_gap = random.randint(1000, 1800)
		else:
			top_offset = 2600
			side_offset = 500
			lead_gap = 1400
		start_y = 0
		for n in range(4):
			if n < 3:
				paper[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset + (n * lead_gap), side_offset + 20 : side_offset + 25] = np.zeros(3) # vertical calibration line left
				paper[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset + (n * lead_gap), side_offset + 220 : side_offset + 225] = np.zeros(3) # vertical calibration line right
				paper[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset-395 + (n * lead_gap), side_offset + 20 : side_offset + 220] = np.zeros(3) # horizontal calibration line top
				paper[start_y + top_offset + (n * lead_gap) : start_y + top_offset + 5 + (n * lead_gap), side_offset : side_offset + 25] = np.zeros(3) # calibration foot left
				paper[start_y + top_offset + (n * lead_gap) : start_y + top_offset + 5 + (n * lead_gap), side_offset + 220 : side_offset + 240] = np.zeros(3) # calibration foot right
				if lead_segmentation:
					label[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset + (n * lead_gap), side_offset + 20 : side_offset + 25] = 12
					label[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset + (n * lead_gap), side_offset + 220 : side_offset + 225] = 12
					label[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset-395 + (n * lead_gap), side_offset + 20 : side_offset + 220] = 12
					label[start_y + top_offset + (n * lead_gap) : start_y + top_offset + 5 + (n * lead_gap), side_offset : side_offset + 25] = 12
					label[start_y + top_offset + (n * lead_gap) : start_y + top_offset + 5 + (n * lead_gap), side_offset + 220 : side_offset + 240] = 12
				if (y[n, 0])>0:
					paper[start_y + top_offset + (n * lead_gap)-int(y[n, 0] * 100000) : start_y + top_offset + 5 + (n * lead_gap), 
						side_offset + 240 : side_offset + 245] = np.zeros(3) # join calibration foot right to trace
				else:
					paper[start_y + top_offset + (n * lead_gap) : start_y + top_offset + (n * lead_gap)-int(y[n, 0] * 100000), 
						side_offset + 240 : side_offset + 245] = np.zeros(3) # join calibration foot right to trace
				paper[start_y + top_offset-int(max(200, max(y[n, 2500] * 100000, y[n + 3, 2500] * 100000))) + (n * lead_gap) : \
					start_y + top_offset + int(max(200, max(-y[n, 2500] * 100000, -y[n + 3, 2500] * 100000))) + (n * lead_gap), 
					side_offset + 240 + 2500 : side_offset + 245 + 2500] = np.zeros(3) # lead divider
				paper[start_y + top_offset-int(max(200, max(y[n + 3, 2500] * 100000, y[n + 6, 2500] * 100000))) + (n * lead_gap)\
					 : start_y + top_offset + int(max(200, max(-y[n + 3, 2500] * 100000, -y[n + 6, 2500] * 100000))) + (n * lead_gap), 
					side_offset + 240 + 5000 : side_offset + 245 + 5000] = np.zeros(3) # lead divider
				paper[start_y + top_offset-int(max(200, y[n + 6, 2500] * 100000)) + (n * lead_gap)\
					 : start_y + top_offset + int(max(200, -y[n + 6, 2500] * 100000)) + (n * lead_gap), 
					side_offset + 240 + 7500 : side_offset + 245 + 7500] = np.zeros(3) # lead divider
			else:
				paper[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset + (n * lead_gap), side_offset + 20 : side_offset + 25] = np.zeros(3)
				paper[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset + (n * lead_gap), side_offset + 220 : side_offset + 225] = np.zeros(3)
				paper[start_y + top_offset-400 + (n * lead_gap) : start_y + top_offset-395 + (n * lead_gap), side_offset + 20 : side_offset + 220] = np.zeros(3)
				paper[start_y + top_offset + (n * lead_gap) : start_y + top_offset + 5 + (n * lead_gap), side_offset : side_offset + 20] = np.zeros(3)
				paper[start_y + top_offset + (n * lead_gap) : start_y + top_offset + 5 + (n * lead_gap), side_offset + 220 : side_offset + 240] = np.zeros(3)
				if (y[rhythm_strip, 0])>0 : 
					paper[start_y + top_offset + (n * lead_gap)-int(y[rhythm_strip, 0] * 100000) : start_y + top_offset + 5 + (n * lead_gap), 
						side_offset + 240 : side_offset + 245] = np.zeros(3) # join calibration foot right to trace
				else:
					paper[start_y + top_offset + (n * lead_gap) : start_y + top_offset + (n * lead_gap)-int(y[rhythm_strip, 0] * 100000), 
						side_offset + 240 : side_offset + 245] = np.zeros(3) # join calibration foot right to trace
		
		# draw leads
		if debugging:
			print('Drawing annotations took', time.perf_counter() - start, 'seconds')
			
		for n in range(4):
			for m in range(3):
				if debugging:
					start = time.perf_counter()
				x = np.arange((2500 * n) + side_offset + 240, (2500 * (n + 1)) + side_offset + 240)
				temp_y = (y[(3 * n) + m, n * 2500 : (n + 1) * 2500] * -100000).astype(int) + top_offset + (lead_gap * m)
				temp_label = y_label[(3 * n) + m, n * 2500 : (n + 1) * 2500]
				x_c, y_c, y_lab = line_drawer(x, temp_y, temp_label, paper_width, paper_height)
				x_c = np.clip(x_c, 0, paper_width - 1)
				y_c = np.clip(y_c, 0, paper_height - 1)
				if debugging:
					print('Drawing line for lead', n * 3 + m, 'took', time.perf_counter() - start, 'seconds')
					start = time.perf_counter()
				paper[y_c, x_c] = np.zeros(3)
				label[y_c, x_c] = y_lab
				if (thicken_trace):
					for g in range(2):
						y_c += 1
						y_c = np.clip(y_c, 0, paper_height - 1)
						for h in range(2):
							x_c += 1
							x_c = np.clip(x_c, 0, paper_width - 1)
							paper[y_c, x_c] = np.zeros(3)
					if debugging:
						print('Thickening trace for lead', n * 3 + m, 'took', time.perf_counter() - start, 'seconds')
						start = time.perf_counter()
		
		# draw rhythm strip
		x = np.arange(side_offset + 240, 10000 + side_offset + 240)
		temp_y = (y[rhythm_strip,  :] * -100000).astype(int) + top_offset + (lead_gap * 3)
		temp_label = y_label[rhythm_strip,  :]
		x_c, y_c, y_lab = line_drawer(x, temp_y, temp_label, paper_width, paper_height)
		x_c = np.clip(x_c, 0, paper_width - 1)
		y_c = np.clip(y_c, 0, paper_height - 1)
		paper[y_c, x_c] = np.zeros(3)
		if not lead_segmentation:
			label[y_c, x_c] = y_lab
		if (thicken_trace):
			for g in range(2):
				y_c += 1
				y_c = np.clip(y_c, 0, paper_height - 1)
				for h in range(2):
					x_c += 1
					x_c = np.clip(x_c, 0, paper_width - 1)
					paper[y_c, x_c] = np.zeros(3)
		if debugging:
			print('Drawing rhythm strip took', time.perf_counter() - start, 'seconds')
			start = time.perf_counter()

		# convert to image
		image_width = 11200 # this is original size; 2910 is A4 paper size
		image_height = 7792 # this is almost original size; 2100 is A4 paper size

		# generate label
		label = create_thickened_label(label, label_width = image_width, label_height = image_height)
		if debugging:
			print('Creating thickened label took', time.perf_counter() - start, 'seconds')
			start = time.perf_counter()
		n_classes = int(np.amax(label)) + 1
		if save_image: 
			if debugging:
				print('Creating labels')
			if print_merged_image_only == False:
				np.save(filename + '_label.npy', label)
			seg_label = np.zeros((label.shape[0], label.shape[1], 3))
			for classes in range(1, n_classes, 1):
				seg_label[:, :, 0] += (label[:, :] == classes) * class_colors[classes][0]
				seg_label[:, :, 1] += (label[:, :] == classes) * class_colors[classes][1]
				seg_label[:, :, 2] += (label[:, :] == classes) * class_colors[classes][2]
			seg_label = cv2.resize(seg_label, (image_width, image_height), interpolation = cv2.INTER_NEAREST)
			seg_label = Image.fromarray(np.uint8(seg_label)).convert("RGBA")
			label = np.uint8(np.ones((label.shape[0], label.shape[1], 3)) * np.expand_dims(label, axis = -1) * (255 // n_classes))
			im_label = cv2.resize(label, (image_width, image_height), interpolation = cv2.INTER_NEAREST)
			im_label = Image.fromarray(label)
			if print_merged_image_only == False:
				im_label.save(filename + '_label.' + image_type, image_type)
			if debugging:
				print('Creating label image took', time.perf_counter() - start, 'seconds')
				start = time.perf_counter()

		# create image
		if debugging:
			print('Paper shape:', paper.shape)
		im = cv2.resize(np.uint8(paper), (image_width, image_height), interpolation = cv2.INTER_AREA)
		im = Image.fromarray(im).convert("RGBA")
		if debugging:
			print('Transposing ECG to image and resizing took', time.perf_counter() - start, 'seconds')
			start = time.perf_counter()
		height_ratio = image_height / paper_height
		width_ratio = image_width / paper_width
		leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
		font = find_font()
		pil_font = ImageFont.truetype(font,  
			size = 108, 
			encoding = "unic")
		draw = ImageDraw.Draw(im)
		black = "#000000"
		x_offset = (side_offset + 300) * width_ratio
		y_offset = (100 + top_offset) * height_ratio
		for lead in range(len(leads)):
			draw.text((x_offset, y_offset), leads[lead], font = pil_font, fill = black)
			y_offset += lead_gap * height_ratio
			if y_offset > (((100 + top_offset) * height_ratio) + (lead_gap * height_ratio * 2)):
				y_offset = (100 + top_offset) * height_ratio
				x_offset += 2500 * width_ratio
		draw.text(((side_offset + 300) * width_ratio, ((top_offset + 100) * height_ratio) + (lead_gap * height_ratio * 3)), 
			'II', 
			font = pil_font, 
			fill = black)
		draw.text(((side_offset) * width_ratio, ((top_offset + 750) * height_ratio) + (1400 * height_ratio * 3)), 
			'200 Hz', 
			font = pil_font, 
			fill = black)
		draw.text(((side_offset + 800) * width_ratio, ((top_offset + 750) * height_ratio) + (1400 * height_ratio * 3)), 
			'25.0 mm/s', 
			font = pil_font, 
			fill = black)
		draw.text(((side_offset + 1800) * width_ratio, ((top_offset + 750) * height_ratio) + (1400 * height_ratio * 3)), 
			'10.0 mm/mV', 
			font = pil_font, 
			fill = black)
		if print_meta_data:
			text_vertical_position = 2000
			draw.text(((side_offset + 600) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				'HR ' + str(meta_data['hr']) + 'bpm', 
				font = pil_font, 
				fill = black)
			text_vertical_position -= 200
			if 'pr' in meta_data.keys() : 
				draw.text(((side_offset + 600) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
					'PR 0.' + str(meta_data['pr']) + 's', 
					font = pil_font, 
					fill = black)
				text_vertical_position -= 200
			draw.text(((side_offset + 600) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				'QRS 0.' + str(meta_data['qrs']) + 's', 
				font = pil_font, 
				fill = black)
			text_vertical_position -= 200
			draw.text(((side_offset + 600) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				'QT/QTc 0.' + str(meta_data['qt']) + 's/0.' + str(meta_data['qtc']) + 's', 
				font = pil_font, 
				fill = black)
			text_vertical_position -= 200
		if random_text: 
			text_offset = random.randint(600, 3000)
			text_vertical_position = random.randint(1000, 2500)
			letters = string.ascii_letters
			txt = ''.join(random.choice(letters) for i in range(random.randint(5, 25)))
			draw.text(((side_offset + text_offset) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				txt, 
				font = pil_font, 
				fill = black)
			text_vertical_position -= 200
			txt = ''.join(random.choice(letters) for i in range(random.randint(5, 25)))
			draw.text(((side_offset + text_offset) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				txt, 
				font = pil_font, 
				fill = black)
			text_vertical_position -= 200
			txt = ''.join(random.choice(letters) for i in range(random.randint(5, 25)))
			draw.text(((side_offset + text_offset) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				txt, 
				font = pil_font, 
				fill = black)
			text_vertical_position -= 200
			txt = ''.join(random.choice(letters) for i in range(random.randint(5, 25)))
			draw.text(((side_offset + text_offset) * width_ratio, (top_offset - text_vertical_position) * height_ratio), 
				txt, 
				font = pil_font, 
				fill = black)
			if debugging:
				print('Writing ECG text took', time.perf_counter() - start, 'seconds')
				start = time.perf_counter()

		if save_image:
			if include_key:
				draw = ImageDraw.Draw(seg_label)
				rect_height_start = 50 * height_ratio
				rect_width_start = 9000 * width_ratio
				rect_diam = 100
				pil_font = ImageFont.truetype(font,  
					size = 48, 
					encoding = "unic")
				for classes in range(1, n_classes, 1):
					rect_height_start += int(rect_diam * 1.5)
					draw.rectangle([rect_width_start, rect_height_start,
								rect_width_start + rect_diam, rect_height_start + rect_diam], 
							fill = "rgb(" + str(class_colors[classes][0]) + "," \
								+ str(class_colors[classes][1]) + "," \
								+ str(class_colors[classes][2]) + ")")
					draw.text((rect_width_start + rect_diam + 30, rect_height_start), 
						segmentation_key[classes], 
						font = pil_font, 
						fill = "#0000FF")
			if print_merged_image_only == False:
				im.save(filename + '_ECG.' + image_type,  image_type)
			edited_image = Image.fromarray(np.uint8(cv2.addWeighted(np.asarray(im), 
				0.5, np.asarray(seg_label), 0.5, 0)))
			if save_dim != 0:
				edited_image = edited_image.resize((save_dim, save_dim), Image.ANTIALIAS)
			if '.' in filename:
				filename = filename.split('.')[0]
			edited_image.save(filename + '_merged_image.' + image_type, image_type)
			if debugging:
				print('Saving ECG and merged image took', time.perf_counter() - start, 'seconds')
				start = time.perf_counter()

		return im, label

def training_generator(
		rhythms=['sr', 'af'], 
		rates=[40, 80], 
		impedance=1., 
		universal_noise_multiplier=1., 
		p_multiplier=1., 
		t_multiplier=1., 
		presets=ecg_presets,
		return_signal=False,
		include_points=True,
		new_random_seed=False,
		segment_leads=False
		):

	if new_random_seed:
		random.seed(time.time())

	ecg_presets = presets
	success = False
	universal_noise_multiplier = random.uniform(0.1, universal_noise_multiplier)
	impedance = random.uniform(0.5, impedance)

	if p_multiplier < 1.:
		p_multiplier = random.uniform(p_multiplier, 1.)
	elif p_multiplier > 1.:
		p_multiplier = random.uniform(1., p_multiplier)
	if t_multiplier < 1.:
		t_multiplier = random.uniform(t_multiplier, 1.)
	elif t_multiplier > 1.:
		t_multiplier = random.uniform(1., t_multiplier)

	preset = random.choice(ecg_presets)
	rhythm = random.choice(rhythms)
	rate = random.randint(rates[0], rates[1])
	print_meta_data = bool(random.getrandbits(1))
	random_text = bool(random.getrandbits(1))
	folder = 'test'
	loop = 1

	if rhythm == 'sr':
		_, _, _, X, y, y_label, meta_data, label_vector = nsr(preset = preset, rate = rate, universal_noise_multiplier = universal_noise_multiplier, impedance = impedance, 
			p_multiplier = p_multiplier, t_multiplier = t_multiplier, include_points = include_points)
	elif rhythm == 'af':
		_, _, _, X, y, y_label, meta_data, label_vector = af(preset = preset, rate = rate, universal_noise_multiplier = universal_noise_multiplier, impedance = impedance, 
			t_multiplier = t_multiplier, include_points = include_points)

	y = evenly_spaced_y(X, y)

	x, y_im = plot_12_lead_ecg(y, y_label, meta_data, filename = folder + str(loop).zfill(4),
		random_layout = True, print_meta_data = print_meta_data, random_text = random_text, save_image = False, 
		lead_segmentation = segment_leads)

	y_lab = np.zeros(6)
	y_lab[presets.index(preset)] = 1

	if return_signal:
		return x, y_im, y
		
	else:
		return x, y_im, y_lab

def training_generator_signal_only(
		rhythms=['sr', 'af'], 
		rates=[40, 80], 
		impedance=1., 
		universal_noise_multiplier=1., 
		p_multiplier=1., 
		t_multiplier=1., 
		presets=ecg_presets,
		return_signal=False,
		include_points=True,
		new_random_seed=False,
		segment_leads=False,
		return_meta_data=False
		):

	if new_random_seed:
		random.seed(time.time())

	ecg_presets = presets
	success = False
	universal_noise_multiplier = random.uniform(0.1, universal_noise_multiplier)
	impedance = random.uniform(0.5, impedance)

	if p_multiplier < 1.:
		p_multiplier = random.uniform(p_multiplier, 1.)
	elif p_multiplier > 1.:
		p_multiplier = random.uniform(1., p_multiplier)
	if t_multiplier < 1.:
		t_multiplier = random.uniform(t_multiplier, 1.)
	elif t_multiplier > 1.:
		t_multiplier = random.uniform(1., t_multiplier)

	preset = random.choice(ecg_presets)
	rhythm = random.choice(rhythms)
	rate = random.randint(rates[0], rates[1])
	print_meta_data = bool(random.getrandbits(1))
	random_text = bool(random.getrandbits(1))
	folder = 'test'
	loop = 1

	if rhythm == 'sr':
		_, _, _, X, y, y_label, meta_data, label_vector = nsr(preset = preset, rate = rate, universal_noise_multiplier = universal_noise_multiplier, impedance = impedance, 
			p_multiplier = p_multiplier, t_multiplier = t_multiplier, include_points = include_points)
	elif rhythm == 'af':
		_, _, _, X, y, y_label, meta_data, label_vector = af(preset = preset, rate = rate, universal_noise_multiplier = universal_noise_multiplier, impedance = impedance, 
			t_multiplier = t_multiplier, include_points = include_points)

	y = evenly_spaced_y(X, y)

	if return_meta_data:

		return y, y_label, meta_data

	else:

		return y, y_label, label_vector

def multiprocess_random_generator(arg_list):

	loop = arg_list[0]
	ecg_presets = arg_list[1]
	rhythms = arg_list[2]

	preset = random.choice(ecg_presets)

	rhythm = random.choice(rhythms)
	rate = random.randint(30, 200)
	print_meta_data = bool(random.getrandbits(1))
	random_text = bool(random.getrandbits(1))

	if rhythm == 'sr':
		_, _, _, X, Y, Y_label, meta_data, label_vector = nsr(
				preset = preset, 
				rate = rate, 
				universal_noise_multiplier = universal_noise_multiplier, 
				impedance = impedance
				)
	elif rhythm == 'af':
		_, _, _, X, Y, Y_label, meta_data, label_vector = af(
				preset = preset, 
				rate = rate, 
				universal_noise_multiplier = universal_noise_multiplier, 
				impedance = impedance
				)

	y = evenly_spaced_y(X, Y)

	plot_12_lead_ecg(
			y, 
			Y_label, 
			meta_data, 
			filename = folder + str(loop).zfill(4), 
			random_layout = True, 
			print_meta_data = print_meta_data, 
			random_text = random_text, 
			include_key = True
			)


if __name__  ==  '__main__': 

	parser = argparse.ArgumentParser(description = 'Argparser')
	parser.add_argument("--rhythm",  type = str,  help = "Specify rhythm (see presets.txt for list)")
	parser.add_argument("--preset",  type = str,  help = "Specify an ECG preset (see presets.txt for list)")
	parser.add_argument("--rate",  type = int,  help = "Specify the heart rate to use with a preset")
	parser.add_argument("--printout",  type = bool,  help = "Specify whether to produce a printed ECG or just the raw signals")
	parser.add_argument("--ecgs",  type = int,  help = "Specify the number of ECGs to generate")
	parser.add_argument("--folder",  type = str,  help = "Specify the folder to save the ECGs to (will overwrite existing ECGs in that folder)")
	parser.add_argument("--noise_multiplier",  type = float,  help = "Specify a mulitplier that will be applied to all noise types")
	parser.add_argument("--impedance",  type = float,  help = "Specify impedance multiplier (higher = greater impedance)")
	parser.add_argument("--multiprocess", type = bool, help = "Specify whether to use multiprocessing (only applicable if using \'random\' preset)")
	parser.add_argument("--cpus", type = int, help = "Number of CPU cores to use if multiprocessing")
	args = parser.parse_args()

	rhythm = args.rhythm
	preset = args.preset
	rate = args.rate
	ecgs = args.ecgs
	folder = args.folder
	universal_noise_multiplier = args.noise_multiplier
	impedance = args.impedance
	use_multiprocessing = args.multiprocess
	cpu_cores = args.cpus

	if ecgs == None:
		ecgs = 1

	if folder == None:
		folder = ''
	else:
		folder = folder + '/'
	Path(folder).mkdir(parents = True,  exist_ok = True)

	if rhythm == None:
		rhythm = 'sr'

	if rate == None:
		rate = 60

	if universal_noise_multiplier == None:
		universal_noise_multiplier = 1.

	if impedance == None:
		impedance = 1.

	if use_multiprocessing == None:
		use_multiprocessing = False

	if cpu_cores == None:
		cpu_cores = 6

	if preset == None:
		print('No preset given')
		preset = ''

		for loop in range(ecgs):
			start = time.perf_counter()

			if rhythm == 'sr':
				_, _, _, X, Y, Y_label, meta_data, label_vector = nsr(
					preset = preset, 
					rate = rate, 
					universal_noise_multiplier = universal_noise_multiplier, 
					impedance = impedance
					)
			elif rhythm == 'af':
				_, _, _, X, Y, Y_label, meta_data, label_vector = af(
					preset = preset, 
					rate = rate, 
					universal_noise_multiplier = universal_noise_multiplier, 
					impedance = impedance
					)
			else:
				print('Error,  rhythm \'' + rhythm + '\' not recognised')
				exit()

			y = evenly_spaced_y(X, Y)

			end = time.perf_counter()

			print("ECG generation took", end - start, "seconds")

			plot_12_lead_ecg(
				y, 
				Y_label, 
				meta_data, 
				filename = folder + str(loop).zfill(4), 
				random_layout = True, 
				include_key = True
				)

			print("ECG image generation took", time.perf_counter() - end, "seconds")

	elif preset == 'random':
		rhythms = ['sr', 'af']

		if use_multiprocessing:

			pool = multiprocessing.Pool(processes = cpu_cores)

			arg_list = []
			for loop in range(ecgs):
				arg_list.append(loop)
				arg_list.append(ecg_presets)
				arg_list.append(rhythms)

			pool.map(multiprocess_random_generator, arg_list)

		else:

			for loop in range(ecgs):

				preset = random.choice(ecg_presets)

				print('Generating ECG signal with preset:', preset + '...')

				rhythm = random.choice(rhythms)
				rate = random.randint(30, 200)
				print_meta_data = bool(random.getrandbits(1))
				random_text = bool(random.getrandbits(1))

				if rhythm == 'sr':
					_, _, _, X, Y, Y_label, meta_data, label_vector = nsr(
						preset = preset, 
						rate = rate, 
						universal_noise_multiplier = universal_noise_multiplier, 
						impedance = impedance
						)
				elif rhythm == 'af':
					_, _, _, X, Y, Y_label, meta_data, label_vector = af(
						preset = preset, 
						rate = rate, 
						universal_noise_multiplier = universal_noise_multiplier, 
						impedance = impedance
						)

				print('Generated signal, plotting to image...')

				y = evenly_spaced_y(X, Y)

				plot_12_lead_ecg(
					y, 
					Y_label, 
					meta_data, 
					filename = folder + str(loop).zfill(4), 
					random_layout = True, 
					print_meta_data = print_meta_data, 
					random_text = random_text, 
					include_key = True
					)

	else:

		if preset in ecg_presets:
			for loop in range(ecgs):				
				print('Using preset\'' + preset + '\'')
				if rhythm == 'sr':
					_, _, _, X, Y, Y_label, meta_data, label_vector = nsr(
						preset = preset, 
						rate = rate, 
						universal_noise_multiplier = universal_noise_multiplier, 
						impedance = impedance
						)
				elif rhythm == 'af':
					_, _, _, X, Y, Y_label, meta_data, label_vector = af(
						preset = preset, 
						rate = rate, 
						universal_noise_multiplier = universal_noise_multiplier, 
						impedance = impedance
						)
				else:
					print('Error,  rhythm \'' + rhythm + '\' not recognised')
					exit()

				print('Label vector', len(label_vector))
				y = evenly_spaced_y(X, Y)

				plot_12_lead_ecg(
					y, 
					Y_label, 
					meta_data, 
					filename = folder + str(loop).zfill(4), 
					random_layout = True, 
					include_key = True
					)

		else:
			print('Preset \'' + preset + '\' not recognised. Please check presets.txt for a full list.')
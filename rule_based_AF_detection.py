import numpy as np
import glob
import cv2
from scipy.signal import find_peaks
from scipy.stats import zscore
from config import config
from helper_functions import class_colors, find_font, n_classes
import multiprocessing
import os
from PIL import Image, ImageDraw, ImageFont

def brute_force_search():

	# segmentation masks were saved in this folder during the validation run:
	folder = config.folder + '/masks'
	files = glob.glob(folder + '/*.0_.npy')
	files_len = len(files)

	best_f1 = 0
	best_p = 0
	best_qrs = 0
	best_std = 0

	# the search ranges were manually set after some experimentation:
	for a in range(10, 20, 1):
		for b in range(7, 12, 1):
			for c in range(7, 14, 1):
				metrics = {
					'tp' : 0,
					'fp' : 0,
					'fn' : 0,
					'tn' : 0,
				}
				
				# iterating through the whole validation set takes ages, so only 100 samples were used:
				for f in range(100):
					y = np.load(files[f])[0,]
					labels = np.array([1 - float(files[f].split('_')[-2])])
					pred = predict_ecg(y, \
						labels, \
						False, \
						p_threshold = a, \
						qrs_with_p_threshold = b, \
						std_threshold = c, \
						use_saved_thresholds = False)[1]

					if labels[0] == 0 and pred == 0:
						metrics['tn'] += 1
					elif labels[0] == 1. and pred == 0:
						metrics['fn'] += 1
					elif labels[0] == 0 and pred == 1:
						metrics['fp'] += 1
					elif labels[0] == 1. and pred == 1:
						metrics['tp'] += 1

				sens = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-8)
				ppv =  metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-8)
				f1 = (2 * sens * ppv) / (sens + ppv + 1e-8)

				if f1 > best_f1:
					best_f1 = f1
					best_p = a
					best_qrs = b
					best_std = c

				print('Done threshold search round', (a - 9) * (b - 6) * (c - 6), \
					'of 100. Current bests:', best_f1, best_p, best_qrs, best_std, end='\r')

			txt = str(best_f1) + ',' + str(best_p) + ',' + str(best_qrs) + ',' + str(best_std)
			with open('brute_force.txt', 'w') as f:
				f.write(txt)

	print('\nDone with threshold search')

def predict_ecg(
		y, 
		labels, 
		test, 
		p_threshold=10, 
		qrs_with_p_threshold=11, 
		std_threshold=8, 
		use_saved_thresholds=True,
		out_path='',
		ecg_path=''
	):

	dim = 768 # hard coded for this demo

	p_threshold = p_threshold
	# DIAGNOSTIC THRESHOLDS FOR AF (set experimentally on synthetic ECGs):
	qrs_with_p_threshold = qrs_with_p_threshold
	std_threshold = std_threshold

	if use_saved_thresholds and os.path.isfile('brute_force.txt'):
		with open('brute_force.txt', 'r') as f:
			thresholds = f.read()
		thresholds = thresholds.split(',')
		p_threshold = int(thresholds[1])
		qrs_with_p_threshold = int(thresholds[2])
		std_threshold = int(thresholds[3])
		
	af = []
	cnt=0

	x = np.empty((y.shape[1], y.shape[2], 3))
	y = np.argmax(y, axis = 0)

	for j in range(3):

		x[:, :, j] = y

	y = cv2.resize(x, (dim, dim), interpolation = cv2.INTER_NEAREST)
	y = y[:, :, 0]

	del x

	# 'lines' is a vector showing the y positions all QRS complexes
	lines = np.sum(y == 3, axis = 1)
	# 'p' is a 2D matrix showing P wave pixels only
	p = y == 1 * 1.
	# 'qrs' is a 2D matrix showing QRS pixels only
	qrs = y == 3 * 1.
	# 'peaks' is a list of the y coordinates of the 4 main ECG lines
	peaks, _ = find_peaks(lines, distance=70)

	# USE A SLIDING WINDOW TO FIND QRS COMPLEXES, LOOK FOR PRECEDING P WAVES
	# AND ESTIMATE R-R DISTANCES
	qrs_with_p = 0
	beat_coords = []
	gaps=[]

	p_sums = ''

	for pk in peaks:

		qrs_line = np.sum(qrs[pk-30 : pk + 30, :], axis = 0)

		# These two lines slide a window across each ECG line looking for discrete QRS complexes and P waves:
		qrs_peaks, _= find_peaks(qrs_line, distance = 15)
		p_line = np.sum(p[pk - 30 : pk + 30, :], axis = 0)

		# This loop checks the area behind each 
		for q in range(len(qrs_peaks)):

			if q > 0:

				gaps.append(qrs_peaks[q] - qrs_peaks[q - 1])

			p_sums += str(np.sum(p_line[qrs_peaks[q] - 30 : qrs_peaks[q]])) + ' '

			if np.sum(p_line[qrs_peaks[q] - 30 : qrs_peaks[q]]) > p_threshold:

				qrs_with_p += 1
				beat_coords.append([pk, qrs_peaks[q]])

	# CALCULATE Z SCORE AND STANDARD DEVIATION OF R-R INTERVALS:
	gaps = np.asarray(gaps)
	z = zscore(gaps)
	gaps_temp = []

	# this loop excludes outliers 
	# (helpful in case a QRS complex has not been segmented - its neighbours will have large R-R gaps):
	for i in range(gaps.size):

		if z[i] < 2:

			gaps_temp.append(gaps[i])

	gaps = np.asarray(gaps_temp)

	std = np.std(gaps)

	# # this code snippet was used to identify threshold search ranges:
	# csv = p_sums + ',' + str(qrs_with_p) + ',' + str(std) + ',' + str(labels[0]) + ',' + str(z) + '\n'
	# if not test:
	# 	with open('rule_based_metrics.csv', 'a+') as f:
	# 		f.write(csv)

	# GENERATE OUTPUT TO EXPLAIN DIAGNOSTIC REASONING:

	if out_path != '':
		if os.path.isfile(ecg_path):
			ecg = np.uint8(np.load(ecg_path))
			ecg = cv2.resize(ecg, (dim, dim), interpolation = cv2.INTER_AREA)[:, :, :3]
			seg_img = np.zeros((dim, dim, 3))
			for c in range(1, n_classes, 1):
				seg_img[:, :, 0] += (y[:, :] == c) * (class_colors[c][0])
				seg_img[:, :, 1] += (y[:, :] == c) * (class_colors[c][1])
				seg_img[:, :, 2] += (y[:, :] == c) * (class_colors[c][2])

			im = Image.fromarray(np.uint8(cv2.addWeighted(ecg, 
								0.5, np.uint8(seg_img), 0.5, 0))).convert('RGBA')

			draw = ImageDraw.Draw(im, mode='RGBA')
			highlight = (255,255,255,100)
			black = "#FF0000"
			font = find_font()
			pil_font = ImageFont.truetype(font, 
				size = 16,
				encoding = "unic")

			im_np = np.asarray(im)
			x_scale = im_np.shape[1] / dim
			y_scale = im_np.shape[0] / dim

			for b in beat_coords:
				x = b[1] * x_scale
				y = b[0] * y_scale
				draw.rectangle([x - 20, y - 20, x + 20, y + 20], outline = black)

			im = im.resize((1024, 713))
			draw = ImageDraw.Draw(im, mode='RGBA')

			diagnosis_text = ''
			if qrs_with_p > p_threshold:
				diagnosis_text = 'sinus rhythm'
			else:
				diagnosis_text = 'atrial fibrillation'

			# draw.text((im_np.shape[0]- (im_np.shape[0] // 5), im_np.shape[1] // 5),
			gap_text = 'There is '
			if std > std_threshold:
				gap_text += 'high'
			else:
				gap_text += 'low'
			gap_text += ' R-R variability.'
			qrs_text = 'The algorithm has detected '+str(qrs_with_p)+' QRS complexes preceded by P waves\n'
			if qrs_with_p > 0:
				qrs_text += '(highlighted by the boxes below). '
			draw.text((50, 20),
				qrs_text + gap_text + '\n' + 
				'On balance, this is likely to be '+ diagnosis_text,
				font = pil_font,
				fill = black)

			im.save(out_path + '.png','PNG')

		else:
			print('Error, no ECG file found!')
	

	if qrs_with_p < qrs_with_p_threshold and std > std_threshold:

		return np.array([0, 1])

	else:

		return np.array([1, 0])


if __name__ == '__main__':

	# brute_force_search()
	idx = 2 # 2 for SR
	files = glob.glob('/media/rbrisk/storage01/ecg_data_enhanced/*_label.npy')
	ecg_path = files[idx].split('_label')[0] + '_ECG.npy'
	ecg = np.load(ecg_path)
	y = np.load(files[idx])
	y = np.moveaxis(np.unpackbits(y).reshape((ecg.shape[0], ecg.shape[0], n_classes + 1)), -1, 0)

	predict_ecg(y, None, None, ecg_path = ecg_path, out_path = 'explanation_SR')
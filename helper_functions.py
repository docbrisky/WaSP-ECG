import numpy as np
from scipy.signal import savgol_filter
import cv2
import random
from config import config
import matplotlib.font_manager
import shutil
# suppress a complex number warning from the savgol_filter
import warnings
warnings.filterwarnings("ignore")

random.seed(config.random_seed)

class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

ecg_presets = [
	'LAHB',
	'LPHB',
	'high_take_off',
	'LBBB',
	'ant_STEMI',
	'random_morphology'
	]

segmentation_key = {
	0 : "background",
	1 : "P wave",
	# 16 : "Biphasic P wave",
	2 : "PR interval",
	3 : "QRS complex",
	# 17 : "Broad QRS with RSR pattern",
	# 18 : "Broad QRS without RSR pattern",
	# 19 : "Inverted narrow QRS complex",
	# 20 : "Inverted QRS with RSR pattern",
	# 21 : "Inverted QRS without RSR pattern",
	4 : "ST segment",
	# 22 : "Upsloping ST segment",
	# 23 : "Downsloping ST segment",
	5 : "T wave",
	# 24 : "Inverted T wave",
	6 : "T-P segment",
	7 : "T/P overlap",
	8 : "start of P",
	9 : "start of Q wave",
	10 : "Q trough",
	11 : "R peak",
	12 : "R prime peak",
	13 : "S trough",
	14 : "J point",
	15 : "end of QT segment (max slope method)",
	16 : "end of T wave (absolute method)"
}

# the classes that represent ECG points rather than waves:
# (note that point 25 - T wave end - is included in waves as it is used to
# create bounding boxes)
points = [8, 9, 10, 11, 12, 13, 14, 15]
# points that should be present for every beat:
vital_points = [9, 11, 14]
# maps points to waves:
point_mapper = {
	8 : 'P wave',
	9 : 'QRS',
	10 : 'QRS',
	11 : 'QRS',
	12 : 'QRS',
	13 : 'QRS',
	14 : 'QRS',
	15 : 5
	}
# the classes that represent ECG waves:
waves = np.setdiff1d([*range(len(segmentation_key))], points).tolist()

n_classes = len(waves) - 1

def evenly_spaced_y(original_x, y): 

	# Don't bother with the spacing function if snippet < 5 mS
	if y.shape[-1] < 5:
		return y

	else:
		# Transform a vector into an array so the function generalises to both:
		if len(original_x.shape) == 1: 
			original_x = np.expand_dims(original_x, axis = 0)
			y = np.expand_dims(y, axis = 0)

		new_x = np.arange(original_x.shape[1])
		intercepts = np.zeros((original_x.shape[0], new_x.shape[0]))

		for lead in range(original_x.shape[0]): 
			np.seterr(divide = 'ignore')
			grads = (y[lead, : -1] - y[lead, 1: ]) / (original_x[lead, : -1] - original_x[lead, 1:])
			placeholder = 0
			for i in range(new_x.shape[0]): 
				for h in range(placeholder, original_x.shape[1], 1): 
					if original_x[lead, h] >= new_x[i]: 
						intercepts[lead, i] = y[lead, h] + ((original_x[lead, h] - new_x[i]) * (-grads[max(h - 1, 0)]))
						placeholder = h
						break

		if intercepts.shape[0] == 1: 
			intercepts = intercepts.reshape(intercepts.shape[1])

		return intercepts

def smooth_and_noise(
		y, 
		rhythm='sr', 
		universal_noise_multiplier=1., 
		impedance=1.
		): 

	# universal_noise_multiplier is a single 'volume control' for all noise types
	y = y * (1 / impedance)

	# generate baseline noise: 
	n = np.zeros((y.size, ), dtype = complex)
	n[40: 100] = np.exp(1j * np.random.uniform(0, np.pi * 2, (60,)))
	atrial_fibrillation_noise = np.fft.ifft(n)
	atrial_fibrillation_noise = savgol_filter(atrial_fibrillation_noise, 31, 2)
	atrial_fibrillation_noise = atrial_fibrillation_noise[: y.size] * random.uniform(.01, .1)
	y = y + (atrial_fibrillation_noise * random.uniform(0, 1.3)  *  universal_noise_multiplier)
	y = savgol_filter(y, 31, 2)

	# generate random electrical noise from leads
	lead_noise = np.random.normal(0, 1 * 10**-5, y.size)

	# generate EMG frequency noise
	emg_noise = np.zeros(0)
	emg_noise_partial = np.sin(np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000) * 10000) * 10**-5
	for r in range(y.size // 1000): 
		emg_noise = np.concatenate((emg_noise, emg_noise_partial))
	emg_noise = np.concatenate((emg_noise, emg_noise_partial[: y.size % 1000]))

	# combine lead and EMG noise, add to ECG
	noise = (emg_noise + lead_noise) * universal_noise_multiplier

	# randomly vary global amplitude
	y = (y + noise) * random.uniform(0.5, 3)

	# add baseline wandering
	skew = np.linspace(0, random.uniform(0, 2) * np.pi, y.size)
	skew = np.sin(skew) * random.uniform(10**-3, 10**-4)
	y = y + skew

	return y

def line_drawer(x, y, y_label, paper_width, paper_height):
 

	# x = a continuous range of pixel columns for a single lead in GE configuration
	# y = the correspoding row value for the ECG line at each pixel column in x 
	# y_label = the class of each pixel (x[i], y[i])

	x_list = x.tolist()
	y_list = y.tolist()
	y_label_list = y_label.tolist()

	diff = y[:-1] - y[1:]
	diff_neg = np.argwhere(diff < -1).tolist()
	diff_pos = np.argwhere(diff > 1).tolist()

	for i in range(len(diff_neg)):
		idx = diff_neg[i][0]
		y_add = [*range(y_list[idx] - 1, y_list[idx + 1], 1)]
		len_y_add = len(y_add)
		y_list += y_add
		x_list += [x_list[idx]] * len_y_add
		y_label_list += [y_label_list[idx]] * len_y_add

	for i in range(len(diff_pos)):
		idx = diff_pos[i][0]
		y_add = [*range(y_list[idx] + 1, y_list[idx + 1], - 1)]
		len_y_add = len(y_add)
		y_list += y_add
		x_list += [x_list[idx]] * len_y_add
		y_label_list += [y_label_list[idx]] * len_y_add

	x = np.array(x_list)
	y = np.array(y_list)
	y_label = np.array(y_label_list)

	return x, y, y_label

def create_thickened_label(label, 
		label_width=2910, 
		label_height=2100, 
		kernel_size=(7, 7),
		iterations=2): 

	kernel = np.ones(kernel_size, np.uint8)
	if config.waves_only:
		label = cv2.dilate(label, kernel, iterations = iterations)
		label = cv2.resize(label, (label_width, label_height), interpolation = cv2.INTER_NEAREST)
	else:
		mask = label
		mask_points = np.where(np.isin(mask, points), mask, 0)
		mask_points = cv2.dilate(mask_points, kernel, iterations = iterations)
		mask_waves = np.where(np.isin(mask, waves), mask, 0)
		mask_waves = cv2.dilate(mask_waves, kernel, iterations = iterations)
		mask_points = cv2.resize(mask_points, (label_width, label_height), interpolation = cv2.INTER_NEAREST)
		mask_waves = cv2.resize(mask_waves, (label_width, label_height), interpolation = cv2.INTER_NEAREST)
		mask = mask_points == 0
		mask_waves = mask_waves * mask
		label = mask_points + mask_waves

	return label

def scale_array(x):

	shape = list(x.shape)
	samples = shape[-1]

	if samples == 10000:
		return x
	elif samples == 1000:
		scale_factor = 10
	elif samples == 2500:
		scale_factor = 4
	elif samples == 5000:
		scale_factor = 2
	else:
		print('Error, target array must contain 1000, 2500, 5000 or 10000 samples per lead!')
		return null

	shape[-1] = shape[-1] * scale_factor
	x = np.reshape(x, -1)

	if x.size % 12 != 0:
		print('Error, target array must contain 12 leads!')
		return null

	y = np.empty_like(x)

	for i in range(0, x.size, samples):
		y[i : i + samples - 1] = x[i + 1 : i + samples]	
		y[i + samples - 1] = x[i + samples - 1]

	a = np.empty(x.size * scale_factor)
	a[:: scale_factor] = x
	diff = (y - x) / scale_factor

	for i in range(1,scale_factor,1):
		z = (diff * i) + x
		a[i :: scale_factor] = z

	a = np.reshape(a, tuple(shape))
	
	return a

def create_paper():

	image = np.ones((7800, 11200, 3)).astype(int)
	image = image * 255
	top_offset = 200
	image[top_offset : -top_offset - 1 : 40, : , 1:] = 0 # horizontal minor lines
	image[top_offset : -top_offset + 1, 40 :: 40, 1:] = 0 # vertical minor lines

	image[top_offset + 1 : -top_offset + 2 : 200, : , 1:] = 0 # horizontal major lines
	image[top_offset + 1 : -top_offset + 2, 201 :: 200, 1:]=0 # vertical major lines

	image[top_offset + 2 : -top_offset + 3 : 200, : , 1:] = 0 # horizontal major lines
	image[top_offset + 2 : -top_offset + 3, 202 :: 200, 1:]=0 # vertical major lines

	np.save('paper.npy',image)

def find_font():

	font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

	for f in font_list:
		if 'times' in f or 'Times' in f:
			return f
		elif 'arial' in f or 'Arial' in f:
			return f
		elif 'courier' in f or 'Courier' in f:
			return f
		elif 'serif' in f or 'Serif' in f:
			return f

	return font_list[0]

def indices_to_one_hot(mask, n_classes, waves_only=False, one_dimensional=False):

	if one_dimensional:

		empty_mask = np.zeros((mask.shape[0], len(segmentation_key)))
				
		for c in range(len(segmentation_key)):
			empty_mask[:, c] = (mask  ==  c)

		if waves_only:
			empty_mask = empty_mask[:, waves]

	else:

		empty_mask = np.zeros((mask.shape[0], mask.shape[1], len(segmentation_key)))
					
		for c in range(len(segmentation_key)):
			empty_mask[:, :, c] = (mask  ==  c)

		if waves_only:
			empty_mask = empty_mask[:, :, waves]

	return empty_mask
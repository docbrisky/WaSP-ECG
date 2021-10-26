# This script generates the wavelets of the ECG
import numpy as np

# P wave generator
# p_length = length of p wave in mS
# p_voltage = magnitude of p wave in mV
# p_biphasic = whether p wave is p_biphasic
# p_biphasic_ratio = magnitude of second peak as a multiple of first peak
# (whichever wave is larger will be normalised to size of p_voltage)
# p_lean = skew of p wave (does not apply to biphasic p waves). < 1 skews left, > 1 skews right
# flipper = voltage multiplier applied uniformly to entire wave (1 has no effect, -1 flips whole wave)
def P_wave(p_length = 100,
		p_voltage = 0.25,
		p_biphasic = False,
		p_biphasic_ratio = 1.5,
		p_lean = 0.8,
		flipper = 1):
	if p_biphasic:
		x = np.linspace(-2.172, 2.172, p_length) ** 2
		y = np.sin(x) * p_voltage
		multiplier = np.linspace(1, p_biphasic_ratio, (p_length * 3) // 4)
		multiplier = np.append(multiplier, np.linspace(p_biphasic_ratio, 1, p_length - multiplier.size))
	else:
		x = np.linspace(-0.5 * np.pi, 1.5 * np.pi,p_length)
		y = ((np.sin(x) + 1) / 2) * p_voltage
		multiplier = np.linspace(1, p_lean, p_length)
	y = y * multiplier
	x = np.linspace(y[0], y[-1], y.shape[0])
	y = y - x
	if p_voltage < 0:
		y = y * (p_voltage / np.amin(y))
	elif p_voltage > 0:
		y = y * (p_voltage / np.amax(y))

	y = y * 0.001 * flipper
	x = np.linspace(0, y.size, y.size)

	return x, y

# Helper function for QRS complex
def sin_wave_generator(start, stop, duration, duration_counter, waves):
	x = np.linspace(np.pi * start, np.pi * stop, duration[duration_counter])
	duration_counter += 1
	return x, duration_counter

# QRS complex generator
# qrs_duration = length of QRS complex in mS
# q_depth = magnitude of Q wave in mV
# q_to_qr_duration_ratio = ratio of Q wave duration to QR complex duration
# r_height = magnitude of R wave in mV
# r_1_upswing_ratio = duration of R wave upswing as a fraction of total R wave duration
# r_1_downswing_ratio = duration of initial R wave downswing as a fraction of total R wave duration
# r_prime_present = boolean denoting presence of R' wave
# r_to_r_prime_duration_ratio = ratio of duration of R : R'
# r_prime_height = magnitude of R' wave in mV
# r_2_upswing_ratio = duration of R' wave upswing as a fraction of total R' wave duration
# r_2_downswing_ratio = duration of initial R' wave downswing as a fraction of total R' wave duration
# s_prime_height = if S wave is "W" shaped, magnitude of positive mid-wave inflection in mV
# s_present = boolean denoting presence of S wave
# s_to_qrs_duration_ratio = ratio of S wave duration to QRS complex duration
# s_depth = magnitude of S wave in mV
# flipper = voltage multiplier applied uniformly to entire wave (1 has no effect, -1 flips whole wave)
# j_point = magnitude of J point in mV
def QRS_complex(qrs_duration = 80,
		q_depth = 0.2,
		q_to_qr_duration_ratio = 0.2,
		r_height = 1,
		r_1_upswing_ratio = 0.5,
		r_1_downswing_ratio = 0.25,
		r_prime_present = True,
		r_to_r_prime_duration_ratio = 1,
		r_prime_height = 1,
		r_2_upswing_ratio = 0.5,
		r_2_downswing_ratio = 0.25,
		s_prime_height = 0.2,
		s_present = True,
		s_to_qrs_duration_ratio = 0.1,
		s_depth = 0.1,
		flipper = 1,
		j_point = 0):

	q_peak = -1
	r_peak = -1
	r_prime_peak = -1
	s_peak = -1

	# The duration of each wavelet is calculated ahead of time and stored as a list
	duration = []
	# The name of each waves is stored for debugging
	waves = []

	if r_height > 0 and r_prime_height < 0 and r_prime_present:
		s_present = False

	# Calculate the duration of the QR complex
	if s_present:
		qr_duration = int(qrs_duration / (1 + s_to_qrs_duration_ratio))
	else:
		qr_duration = qrs_duration

	# Calculate duration of Q wave (if present)
	if q_depth > 0:
		q_length = int(q_to_qr_duration_ratio * qr_duration)
		duration.append(q_length)
		qr_duration -= q_length
		waves.append('Q')

	# Calculate duration of R
	if r_prime_present:
		r1_duration = int(qr_duration / (1 + r_to_r_prime_duration_ratio))
	else:
		r1_duration = qr_duration

	# Both R and R prime (if present) comprise 3 parts:
	# 1. The upslope of the wave
	# 2. The initial downslope of the wave, whose gradient increases over time
	# 3. The final downslope of the wave, whose gradient decreases over time

	# Calculate the duration of the 3 parts of R
	r1_us = int(r1_duration * r_1_upswing_ratio)
	r1_ds = int(r1_duration * r_1_downswing_ratio)
	duration.append(r1_us)
	waves.append('R1_1')
	duration.append(r1_ds)
	waves.append('R1_2')
	if (r_prime_present  ==  False and s_present  ==  False):
		duration.append(qrs_duration - sum(duration))
		waves.append('R1_3')
	else:
		duration.append(r1_duration - r1_us - r1_ds)
		waves.append('R1_3')

	# Calculate the duration of the 3 parts of R'
	if r_prime_present:
		r2_duration = qr_duration - r1_duration
		r2_us = int(r2_duration * r_2_upswing_ratio)
		r2_ds = int(r2_duration * r_2_downswing_ratio)
		duration.append(r2_us)
		waves.append('R2_1')
		duration.append(r2_ds)
		waves.append('R2_2')
		if (s_present == False):
			duration.append(qrs_duration-sum(duration))
			waves.append('R2_3')
		else:
			duration.append(r2_duration - r2_us - r2_ds)
			waves.append('R2_3')

	# Calculate duration of S
	if s_present:
		duration.append(qrs_duration-sum(duration))
		waves.append('S')
	
	# Set a placeholder to track sample number 
	duration_counter = 0

	# Calculate voltage of wavelets:
	# Q wave
	if q_depth > 0:
		x, duration_counter = sin_wave_generator(0.5, 1.5, duration, duration_counter, waves)
		q = ((np.sin(x) - 1) / 2) * q_depth
		q_peak = np.argmin(q)
	else:
		q = np.zeros(0)

	# QR wavelet
	x, duration_counter = sin_wave_generator(-0.5,0, duration, duration_counter, waves)
	qr = (((np.sin(x) + 1)) * (r_height + q_depth)) - q_depth
	if (not r_prime_present \
			or (s_prime_height <= r_height) \
			or (r_height > 0 and s_prime_height < 0)) \
		and (not (r_prime_height > r_height \
			and s_prime_height > r_height)):
		r_peak = q.size + qr.size

	# RS wave
	x, duration_counter = sin_wave_generator(0.5, 1, duration, duration_counter, waves)
	if r_prime_present:
		# rs1 is first half of downwards inflection after R peak
		rs1 = (((np.sin(x) + 1) / 2) * (r_height - s_prime_height)) + s_prime_height
		x, duration_counter = sin_wave_generator(1, 1.5, duration, duration_counter, waves)
		# rs2 is second half of downwards inflection after R peak
		rs2 = (((np.sin(x) + 1) / 2) * (r_height - s_prime_height)) + s_prime_height
		rs = np.concatenate((rs1, rs2[1:]))
		if s_prime_height > r_height and s_prime_height > r_prime_height:
			r_peak = q.size + qr.size +  rs.size
		elif r_height > 0 and r_prime_height < 0 \
			and(abs(s_prime_height - r_prime_height) <= abs(r_prime_height) / 10):
			s_peak = q.size + qr.size +  rs.size
		x, duration_counter = sin_wave_generator(-0.5, 0.5, duration,duration_counter, waves)
		# sr is upwards inflect from S prime towards R prime
		sr = (((np.sin(x) + 1) / 2) * (r_prime_height - s_prime_height)) + s_prime_height
		if (r_height >= s_prime_height and (r_prime_height > s_prime_height or abs(s_prime_height - r_prime_height) <= abs(r_prime_height) / 10)) \
			or (r_height > 0 and r_prime_height < 0 and (abs(s_prime_height - r_prime_height) <= abs(r_prime_height) / 10)):
			r_prime_peak = q.size + qr.size + rs.size + sr.size
		elif r_height > 0 and r_prime_height < 0 and (abs(s_prime_height - r_prime_height) > abs(r_prime_height)):
			s_peak = q.size + qr.size + rs.size + sr.size
		elif r_prime_height > r_height and r_prime_height > s_prime_height:
			r_peak = q.size + qr.size + rs.size + sr.size
		x, duration_counter = sin_wave_generator(0.5, 1, duration, duration_counter, waves)
		# rs1 is first half of downwards inflection after R prime peak
		rs3 = (((np.sin(x) + 1) / 2) * (r_prime_height))
		x, duration_counter = sin_wave_generator(1, 1.5, duration, duration_counter, waves)
		rs1 = np.concatenate((rs,sr,rs3))
		if s_present:
			# rs2 is second half of downwards inflection after R prime peak (to S trough)
			rs2 = (((np.sin(x) + 1) / 2) * (r_prime_height + (s_depth * 2))) - s_depth
			rs = np.concatenate((rs1, rs2[1:]))
			s_peak = q.size + qr.size +  rs.size
			x, duration_counter = sin_wave_generator(-0.5, 0.5, duration, duration_counter, waves)
			# s is upwards inflection from S trough to J point
			s = (((np.sin(x) + 1) / 2) * (s_depth + j_point)) - s_depth
			y = np.concatenate((q, qr, rs, s))
		else:
			s_depth = 0
			# rs2 is second half of downwards inflection after R peak prime to J point
			rs2 = ((np.sin(x) + 1) / 2) * (r_prime_height - (j_point * 2)) + j_point
			rs = np.concatenate((rs1, rs2[1:]))
			y = np.concatenate((q, qr, rs))
	else:
		# rs1 is first half of downwards inflection after R peak
		rs1 = (((np.sin(x) + 1) / 2) * (r_height))
		x, duration_counter = sin_wave_generator(1.25, 1.5, duration, duration_counter, waves)
		if s_present:
			# rs2 is second half of downwards inflection after R prime peak (to S trough)
			rs2 = (((np.sin(x) + 1)) * (r_height + s_depth)) - s_depth
			rs = np.concatenate((rs1, rs2[1:]))
			s_peak = q.size + qr.size + rs.size
			x, duration_counter = sin_wave_generator(-0.5, 0.5, duration, duration_counter, waves)
			# s is upwards inflection from S trough to J point
			s = ((np.sin(x) / 2) + 0.5) * (s_depth + j_point) - s_depth
			y = np.concatenate((q, qr, rs, s))
		else:
			s_depth = 0
			# rs2 is second half of downwards inflection after R peak to J point
			rs2 = ((np.sin(x) + 1)) * (r_height - (j_point * 2)) + (j_point)
			rs = np.concatenate((rs1, rs2[1:]))
			y = np.concatenate((q, qr, rs))

	# Skew the QRS complex slightly to make it more realistic
	x = np.linspace(0, y.size, y.size)
	x_log = np.logspace(1, 2, num = rs2.size - 1)
	x_log = (x_log / np.amax(x_log)) * (rs2.size - 1)
	x_norm = np.linspace(x_log[0], 0, x_log.size)
	x_log = x_log - x_norm
	x[q.size + qr.size + rs1.size : q.size + qr.size + rs.size] = x_log + q.size + qr.size + rs1.size

	# Flip and scale voltages
	y = y * flipper * 0.001

	wave_peak_list = [q_peak, r_peak, r_prime_peak, s_peak]

	return x, y, wave_peak_list

# ST segment generator
# j_point = magnitude of J point in mV
# st_delta = change in magnitude of ST segment over the course of the wave in mV (0.1 slopes up, -0.1 slopes down)
# st_length = duration of ST segment in mS
# t_height = not used
# flipper = voltage multiplier applied uniformly to entire wave (1 has no effect, -1 flips whole wave)
def ST_segment(j_point = 0,
		st_delta = 0,
		st_length = 50,
		t_height = 0.5,
		flipper = 1):
	if (st_delta < 0):
		x = np.linspace(0.5 * np.pi, 1.75 * np.pi, st_length)
		y = ((np.sin(x) + 1) / 2) * -st_delta
	else:
		x = np.linspace(-0.5 * np.pi, 0.5 * np.pi, st_length)
		y = ((np.sin(x) + 1) / 2) * st_delta
	y = y + j_point

	x = np.linspace(0, (j_point + st_delta) - y[-1], st_length)
	y = y + x

	y = y * flipper * 0.001
	x = np.linspace(0, st_length, st_length)

	# for no curve:
	y = np.linspace(j_point, j_point + st_delta, st_length)
	y = y * 0.001 * flipper
	x = np.linspace(0, y.size, y.size)

	return x, y

# T wave generator
# st_end = magnitude of the end of the ST segment in mV
# t_height = magnitude of the T wave in mV
# t_length = duration of the T wave in mS
# flipper = voltage multiplier applied uniformly to entire wave (1 has no effect, -1 flips whole wave)
# t_lean = variable that skews T wave in order to increase variation in morphology
def T_wave(st_end = 0,
		t_height = 0.5,
		t_length = 150,
		flipper = 1,
		t_lean = 1):

	x = np.linspace(-np.pi * 0.5, np.pi * 1.5, t_length)
	y = ((np.sin(x) + 1) / 2) * (t_height - (st_end / 2))

	angler = np.linspace(st_end, 0, t_length)

	if (t_lean > 0):
		angler = np.flip(angler)
		y = y + angler
		if (t_height > st_end):
			for a in range(y.size // 3):
				if y[-a] < st_end:
					y[-a] = st_end
	else:
		droppy_t_switch = False
		y = y + angler
		last_downslope = 0
		if (t_height > st_end):
			for a in range(y.size // 3):
				if y[a] < st_end:
					y[a] = st_end
					droppy_t_switch = True
		if droppy_t_switch:
			gradient = (y[y.size // 2] - y[0]) / (y.size // 2)
			grad = np.array([gradient])
			y_grad = np.gradient(y)
			idx  =  (np.abs(y_grad - grad)).argmin()
			x = np.linspace(y[0], y[idx], idx)
			y[:idx] = x

	y = y * flipper * 0.001

	if t_lean != 0:
		x = np.logspace(1, 1 + t_lean, num = y.size)
		x = (x / np.amax(x)) * y.size
		x = np.amax(x) - x

		if t_lean > 0:
			x = np.flip(x)
			y = np.flip(y)
	else:
		x = np.linspace(0, y.size, y.size)

	y[-1] = 0

	return x, y
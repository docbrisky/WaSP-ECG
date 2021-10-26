import pandas as pd
import numpy as np
import wfdb
import ast
from helper_functions import scale_array
from ecg_generator import plot_12_lead_ecg
import pickle
import os
import random
from collections import Counter
from config import config
random.seed(config.random_seed)

class Physionet_Helper():

	def __init__(self, 
		path='./ptb-xl/', 
		sampling_rate=100, 
		dict_path="diag_list.p", 
		diag_list = ['SR', 'AFIB'],
		superclasses = False
		):

		self.path = path
		self.sampling_rate=sampling_rate
		self.superclasses = config.ptb_superclasses
		self.Y = pd.read_csv(self.path+'ptbxl_database.csv', index_col='ecg_id')
		self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: ast.literal_eval(x))
		self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(self.aggregate_diagnostic)

		self.diag_list = config.ptb_diagnoses #  ['NORM', 'MI', 'STTC', 'HYP', 'CD']

		self.n_classes = len(self.diag_list)

		self.train_ids = None

	def aggregate_diagnostic(self, y_dic):

		tmp = []

		agg_df = pd.read_csv(self.path+'scp_statements.csv', index_col=0)
		if self.superclasses:
			agg_df = agg_df[agg_df.diagnostic == 1]
		for key in y_dic.keys():
			if key in agg_df.index:
				if self.superclasses:
					tmp.append(agg_df.loc[key].diagnostic_class)
				else:
					tmp.append(key)

		return list(set(tmp))

	def get_train_list(self, samples_per_class=2000):

		if self.train_ids == None:

			if os.path.isfile('pickles/ptb_train.pickle') and \
				os.path.isfile('pickles/ptb_val.pickle') and \
				os.path.isfile('pickles/ptb_test.pickle'):

				self.train_ids = pickle.load(open('pickles/ptb_train.pickle', 'rb'))
				self.val_ids = pickle.load(open('pickles/ptb_val.pickle', 'rb'))
				self.test_ids = pickle.load(open('pickles/ptb_test.pickle', 'rb'))

			else:

				patient_ids = []
				for index, row in self.Y.iterrows():
					if bool(set(row.diagnostic_superclass) & set(self.diag_list)):
						patient_ids.append([index])

				random.shuffle(patient_ids)

				fifth = len(patient_ids) // 5

				self.train_ids = patient_ids[fifth * 2 :]
				self.val_ids = patient_ids[: fifth]
				self.test_ids = patient_ids[fifth : fifth * 2]

				if not os.path.isdir('pickles'):
					os.mkdir('pickles')

				pickle.dump(self.train_ids, open('pickles/ptb_train.pickle', "wb"))
				pickle.dump(self.val_ids, open('pickles/ptb_val.pickle', "wb"))
				pickle.dump(self.test_ids, open('pickles/ptb_test.pickle', "wb"))

		return self.train_ids

	def get_val_list(self):

		if self.val_ids == None:

			return []

		else:

			return self.val_ids

	def get_test_list(self):

		if self.test_ids == None:

			return []

		else:

			return self.test_ids

	# load and convert annotation data
	def load_sample_ecg(self, pt_id, plot=True):

		row = self.Y.loc[pt_id]
		X = wfdb.rdsamp(self.path + row.filename_hr.values[0])
		X = np.array(X[0])
		X = np.moveaxis(X,-1,0)
		X = scale_array(X)
		X = X * 2e-3

		y = np.zeros(self.n_classes)
		for dsc in row.diagnostic_superclass.values[0]:
			if dsc in self.diag_list:
				y[self.diag_list.index(dsc)] = 1

		if plot:

			X, _ = plot_12_lead_ecg(X, np.zeros_like(X), None, print_meta_data = False, save_image = False)

		return X, y

	def get_n_classes(self):

		return self.n_classes

	def get_diag_list(self):

		return self.diag_list

	def get_all_ids(self):

		patient_ids = []
		for index, row in self.Y.iterrows():
			patient_ids.append([index])
		
		return patient_ids

	def balance_samples(self, ids, target_class=0):

		list_1 = []
		list_2 = []

		for idx in range(len(ids)):

			row = self.Y.loc[ids[idx]]
			hit = False

			for dsc in row.diagnostic_superclass.values[0]:
				if dsc in self.diag_list:
					if self.diag_list.index(dsc) == target_class:
						hit = True
			if hit:
				list_1.append(ids[idx])
			else:
				list_2.append(ids[idx])

		random.shuffle(list_1)
		random.shuffle(list_2)
		if len(list_1) > len(list_2):
			while len(list_1) > len(list_2):
				list_2 += list_2
			list_2 = list_2[:len(list_1)]
		else:
			while len(list_2) > len(list_1):
				list_1 += list_1
			list_1 = list_1[:len(list_2)]

		final_list = list_1 + list_2
		random.shuffle(final_list)

		return(final_list)


if __name__ == '__main__':

	ph = Physionet_Helper()
	diag_list = ph.get_diag_list()
	print(diag_list)
	train_list = ph.get_train_list()
	print(len(train_list))
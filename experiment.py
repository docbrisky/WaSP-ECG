from trainer import train
from config import config
from validator import validate
import shutil
from data_gen import generate_data
from create_ptb_images import create_ptb_images, reset_PTB
import pickle
import os

def train_validate_test(dim_text='1d', model_text='sr_af', test=False, enhanced=False):

	generate_data()
	create_ptb_images()

	# if not enhanced:

	# 	if not os.path.isfile(config.model_folder + '/' + 'unet_' + dim_text + '_' + model_text + '_naive.p'):
	# 		train(
	# 			training_task='classification', 
	# 			model_to_load=['', '', ''],
	# 			model_to_save=['unet_' + dim_text + '_' + model_text + '_naive.p', '', ''],
	# 			file_suffix='_naive_'
	# 			)

	# 	file_suffix = '_naive_val_'
	# 	if not os.path.isfile(config.results_folder + '/' + \
	# 					"validation_results_" + file_suffix + ".csv"):
	# 		validate(
	# 			training_task='classification', 
	# 			model_to_load=['unet_' + dim_text + '_' + model_text + '_naive.p', '', ''],
	# 			file_suffix=file_suffix
	# 			)

	# 	file_suffix = '_naive_test_'
	# 	if test and not os.path.isfile(config.results_folder + '/' + \
	# 					"test_results_" + file_suffix + ".csv"):
	# 		validate(
	# 			training_task='classification', 
	# 			model_to_load=['unet_' + dim_text + '_' + model_text + '_naive.p', '', ''],
	# 			file_suffix=file_suffix,
	# 			test = True
	# 			)

	# 	if dim_text == '2d':

	# 		if not os.path.isfile(config.model_folder + '/' + 'unet_' + dim_text + '_' + model_text + '_completely_naive.p'):
	# 			train(
	# 				training_task='classification', 
	# 				model_to_load=['', '', ''],
	# 				model_to_save=['unet_' + dim_text + '_' + model_text + '_completely_naive.p', '', ''],
	# 				file_suffix='_completely_naive_',
	# 				completely_naive=True
	# 				)

	# 		file_suffix = '_completely_naive_val_'
	# 		if not os.path.isfile(config.results_folder + '/' + \
	# 						"validation_results_" + file_suffix + ".csv"):
	# 			validate(
	# 				training_task='classification', 
	# 				model_to_load=['unet_' + dim_text + '_' + model_text + '_completely_naive.p', '', ''],
	# 				file_suffix=file_suffix,
	# 				completely_naive=True
	# 				)

	# 		file_suffix = '_completely_naive_test_'
	# 		if test and not os.path.isfile(config.results_folder + '/' + \
	# 						"test_results_" + file_suffix + ".csv"):
	# 			validate(
	# 				training_task='classification', 
	# 				model_to_load=['unet_' + dim_text + '_' + model_text + '_completely_naive.p', '', ''],
	# 				file_suffix=file_suffix,
	# 				test = True,
	# 				completely_naive=True
	# 				)

	enhanced_text = ''
	if enhanced:
		enhanced_text += 'enh_'
		model_text += '_enh'

	if not os.path.exists(config.model_folder + '/' + 'unet_' + dim_text + '.p'):
		if enhanced:
			print('Error, enhanced training only works if pretrained model is available')
			print('No model found at ', config.model_folder + '/' + 'unet_' + dim_text + '.p')
			quit()
		else:
			train(
				training_task='segmentation', 
				model_to_load=['', '', ''],
				model_to_save=['unet_' + dim_text + '.p', '', ''],
				file_suffix='_pretrain_'
				)

	if enhanced and not os.path.isfile(config.model_folder + '/' + 'unet_' + enhanced_text + dim_text + '_' + model_text  + '.p'):
		print('Starting multitask training...')
		train(
			training_task='multitask', 
			model_to_load=['unet_' + dim_text + '.p', '', ''],
			model_to_save=['unet_' + enhanced_text + dim_text + '_' + model_text  + '.p', \
				'unet_' + enhanced_text + dim_text + '_' + model_text + '_prelim_classification.p', ''],
			file_suffix='_pretrain_' + enhanced_text
			)

	if not enhanced and \
			not os.path.isfile(config.model_folder + '/' + 'unet_' + enhanced_text + dim_text + '_' + model_text + '_classification.p'):
		train(
			training_task='classification', 
			model_to_load=['unet_' + dim_text  + '.p', '', ''],
			model_to_save=['unet_' + dim_text + '_' + model_text + '_classification.p', '', ''],
			file_suffix='_finetune_'
			)

	elif enhanced and \
			not os.path.isfile(config.model_folder + '/' + 'unet_' + enhanced_text + dim_text + '_' + model_text + '_classification.p'):
		if dim_text == '1d':
			train(
				training_task='classification', 
				model_to_load=['unet_' + enhanced_text + dim_text + '_' + model_text  + '.p', \
					'unet_' + enhanced_text + dim_text + '_' + model_text + '_prelim_classification.p', ''],
				model_to_save=['unet_' + enhanced_text + dim_text + '_' + model_text + '_classification.p', '', ''],
				file_suffix='_finetune_' + enhanced_text
				)
		else:
			train(
				training_task='classification', 
				model_to_load=['unet_' + enhanced_text + dim_text  + '.p', '', ''],
				model_to_save=['unet_' + enhanced_text + dim_text + '_' + model_text + '_classification.p', '', ''],
				file_suffix='_finetune_' + enhanced_text
				)

	file_suffix = '_val_' + enhanced_text
	if not os.path.isfile(config.results_folder + '/' + \
					"validation_results_" + file_suffix + ".csv"):
		validate(
			training_task='classification', 
			model_to_load=['unet_' + enhanced_text + dim_text + '_' + model_text + '_classification.p', '', ''],
			file_suffix=file_suffix
			)

	file_suffix = '_test_'  + enhanced_text
	if test and not os.path.isfile(config.results_folder + '/' + \
					"test_results_" + file_suffix + ".csv"):
		validate(
			training_task='classification', 
			model_to_load=['unet_' + enhanced_text + dim_text + '_' + model_text + '_classification.p', '', ''],
			file_suffix=file_suffix,
			test = True
			)

def run_stages(stage=1, diagnoses='SR_AF', test=True):

	if stage == 1:

		print('Starting stage 1: standard 1D model with diagnoses = ' + diagnoses)
		train_validate_test(test=test, model_text=diagnoses)
		stage = 2
		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		shutil.copyfile('experiments/'+ diagnoses + '_2D.json', 'config.json')
		print('Finished stage 1')
		quit()

	elif stage == 2:

		print('Starting stage 2: standard 2D model with diagnoses = ' + diagnoses)
		train_validate_test(dim_text='2d', test=test, model_text=diagnoses)
		stage = 3
		quit()
		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		shutil.copyfile('experiments/'+ diagnoses + '_1D_enhanced.json', 'config.json')
		print('Finished stage 2')
		quit()

	elif stage == 3:

		print('Starting stage 3: enhanced 1D pretraining with diagnoses = ', diagnoses)
		train_validate_test(test=test, enhanced=True, model_text=diagnoses)
		stage = 4
		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		shutil.copyfile('experiments/'+ diagnoses + '_2D_enhanced.json', 'config.json')
		print('Finished stage 3')
		quit()

	elif stage == 4:

		print('Starting stage 4: enhanced 2D pretraining with diagnoses = ', diagnoses)
		train_validate_test(dim_text='2d', test=test, model_text=diagnoses, enhanced=True)
		stage = 5
		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		shutil.copyfile('experiments/'+ diagnoses + '_mixed.json', 'config.json')
		print('Finished stage 4')
		quit()

	elif stage == 5:

		print('Starting stage 5: mixed modality model with diagnoses = ', diagnoses)

		dim_text = '2d'
		if not os.path.isfile(config.model_folder + '/' + 'unet_2d_' + diagnoses + '_mixed_classification.p'):
			train(
				training_task='classification', 
				model_to_load=['unet_2d.p', '', ''],
				model_to_save=['unet_2d' + '_' + diagnoses + '_mixed_classification.p', '', ''],
				file_suffix='_finetune_mixed',
				mix_dimensions=True
				)

		file_suffix = '_finetune_mixed_val'
		if not os.path.isfile(config.results_folder + '/' + \
					"validation_results_" + file_suffix + ".csv"):
			validate(
				training_task='classification', 
				model_to_load=['unet_2d.p', 'unet_2d' + '_' + diagnoses + '_mixed_classification.p', ''],
				file_suffix=file_suffix,
				mix_dimensions=True
				)

		file_suffix = '_finetune_mixed_test'
		if test and not os.path.isfile(config.results_folder + '/' + \
						"test_results_" + file_suffix + ".csv"):
			validate(
				training_task='classification', 
				model_to_load=['unet_2d.p', 'unet_2d' + '_' + diagnoses + '_mixed_classification.p', ''],
				file_suffix=file_suffix,
				mix_dimensions=True,
				test = True
				)

		stage = 6
		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		if diagnoses == 'SR_AF':
			shutil.copyfile('experiments/SR_AF_2D.json', 'config.json')
		print('Finished stage 5')

if __name__ == '__main__':

	test = True

	if not os.path.isdir('results'):
		os.mkdir('results')

	if not os.path.isfile('diagnoses.pickle'):
		diagnoses = 'SR_AF'
		with open('diagnoses.pickle', 'wb') as f:
			pickle.dump(diagnoses, f)
	else:
		with open('diagnoses.pickle', 'rb') as f:
			diagnoses = pickle.load(f)

	if not os.path.isfile('stage.pickle'):
		shutil.copyfile('experiments/SR_AF_1D.json', 'config.json')
		stage = 1
		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		print('Experiment set up. Run \'python experiment.py\' again to begin.')
		quit()
	else:
		with open('stage.pickle', 'rb') as f:
			stage = pickle.load(f)

	if stage == 6 and diagnoses == 'SR_AF':

		print('Starting stage 6: rule-based AF detection')

		file_suffix = '_val_rules_'

		if not os.path.isfile(config.results_folder + '/' + \
				"validation_results_" + file_suffix + ".csv"):

			validate(
				training_task='rule_based', 
				model_to_load=['unet_2d.p', '', ''],
				file_suffix=file_suffix
				)

		file_suffix = '_test_rules_'

		if test and not os.path.isfile(config.results_folder + '/' + \
				"test_results_" + file_suffix + ".csv"):

			validate(
				training_task='rule_based', 
				model_to_load=['unet_2d.p', '', ''],
				file_suffix=file_suffix,
				test=True
				)

		stage = 7
		diagnoses = 'NORM_MI'

		with open('stage.pickle', 'wb') as f:
			pickle.dump(stage, f)
		with open('diagnoses.pickle', 'wb') as f:
			pickle.dump(diagnoses, f)
		shutil.copyfile('experiments/NORM_MI_1D.json', 'config.json')
		print('Finished stage 6')
		quit()

	elif stage == 6 and diagnoses == 'NORM_MI':

		stage = 8
		with open('stage.pickle', 'wb') as f:
			 pickle.dump(stage, f)
		os.unlink('diagnoses.pickle')
		print('Finished experiment!')
		quit()

	elif stage == 7:

		print('Preparing for switch to MI detection...')
		reset_PTB()
		stage = 1
		run_stages(stage=stage, diagnoses=diagnoses, test=test)

	elif stage == 8:

		quit()

	else:

		run_stages(stage=stage, diagnoses=diagnoses, test=test)
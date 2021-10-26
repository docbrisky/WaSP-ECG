import pickle
import shutil
from create_ptb_images import create_ptb_images, reset_PTB
from trainer import train
import glob
from config import config
import random
import time
import numpy as np
import cv2
from ptb_loader import Physionet_Helper
import random
from ecg_generator import plot_12_lead_ecg, training_generator_signal_only

X, y, meta_data = training_generator_signal_only(return_meta_data=True)
plot_12_lead_ecg(X, y, meta_data, print_merged_image_only=False, include_key=True)


# stage = 2
# with open('stage.pickle', 'wb') as f:
# 	pickle.dump(stage, f)
# shutil.copyfile('experiments/SR_AF_2D.json', 'config.json')

# train(
# 	training_task='segmentation', 
# 	model_to_load=['unet_2d.p', '', ''],
# 	model_to_save=['unet_2d_augmented.p', '', ''],
# 	file_suffix='_pretrain_augmented_'
# 	)

# ph = Physionet_Helper()
# train_list = ph.get_train_list()
# for i in range(9):
# 	rndx = random.randint(0, len(train_list) - 1)
# 	X, y = ph.load_sample_ecg(train_list[rndx])
# 	X.save('real_ecg' + str(i) + '.png', 'PNG')


# files = glob.glob(config.ptb_folder + '/*ECG.npy')
# for i in range(10):
# 	random.seed(time.time())
# 	randint = random.randint(0, len(files) - 1)
# 	x = cv2.cvtColor(np.load(files[randint]), cv2.COLOR_BGR2RGB)
# 	cv2.imwrite('real_ECG_' + str(i).zfill(2) + '.jpg', x)
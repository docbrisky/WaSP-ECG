import albumentations as A
import numpy as np
import cv2
import random
from helper_functions import indices_to_one_hot, points, waves
import matplotlib.pyplot as plt
import time
from config import config
random.seed(config.random_seed)

class Image_Augmentor():

	def __init__(
			self, 
			dim=2048, 
			n_classes=16, 
			augment=False,
			waves_only=False
			):

		half_dim = int(dim // 2)

		self.n_classes = n_classes
		self.dim = dim
		self.augment = augment
		self.waves_only = waves_only
		self.transform = A.Compose([
			A.HorizontalFlip(p = 0.2),
		    A.RandomBrightnessContrast(p = 0.2),  
		    A.RandomGamma(p = 0.2),    
		    A.CLAHE(p = 0.2),
		    A.HueSaturationValue(hue_shift_limit = 20, 
		    	sat_shift_limit = 50, 
		    	val_shift_limit = 50, 
		    	p = 0.1),
		    A.ChannelShuffle(p = 0.1),
		    A.ShiftScaleRotate(p = 0.2),
			A.RGBShift(p = 0.2),
			A.Blur(p = 0.2),
			A.GaussNoise(p = 0.2),
			A.ElasticTransform(p = 0.2),
			A.Cutout(p = 0.2)
		])

	def transform_image(self, image):

		if isinstance(image, (np.ndarray, np.generic)) == False:
			image = np.asarray(image)

		image = image[:,:,:3]

		if image.shape[0] != self.dim and image.shape[1] != self.dim:
			image = cv2.resize(image, (self.dim, self.dim), interpolation = cv2.INTER_AREA)

		if self.augment:
			image = self.transform(image=image)["image"]

		return image

	def transform_image_mask(self, image, mask):

		if isinstance(image, (np.ndarray, np.generic)) == False:
			image=np.asarray(image)

		image = np.uint8(image[:,:,:3])

		if image.shape[0] != self.dim and image.shape[1] != self.dim:
			image = cv2.resize(image, (self.dim, self.dim), interpolation = cv2.INTER_AREA)

		if mask.shape[0] != self.dim and mask.shape[1] != self.dim:
			mask = cv2.resize(mask, (self.dim, self.dim), interpolation = cv2.INTER_NEAREST)

		if self.augment:
			if len(mask.shape) == 3:
				mask = np.argmax(mask, axis = -1)
			empty_mask = np.empty((mask.shape[0], mask.shape[1], 3))
			for i in range(3):
				empty_mask[:, :, i] = mask
			mask = empty_mask

			transformed = self.transform(image=image, mask=mask)
			image = transformed['image']
			mask = transformed['mask']
			mask = indices_to_one_hot(mask[:, :, 0], self.n_classes, waves_only = self.waves_only)

		elif len(mask.shape) == 2:
			mask = indices_to_one_hot(mask, self.n_classes, waves_only = self.waves_only)

		return image, mask.astype(np.uint8)
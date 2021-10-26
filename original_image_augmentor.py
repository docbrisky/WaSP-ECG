from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import imutils
import math
from config import config
from helper_functions import indices_to_one_hot, n_classes

def add_differential_lighting(image,mask):
	if np.amax(image)<=1.:
		image=(image*255).astype(int)
	image=np.uint8(image)

	im = Image.new('RGB', (image.shape[1], image.shape[0]))
	lighting = im.load()

	heatmap=[]
	for i in range(10):
		l=random.uniform(0,1)
		heatmap.append([random.uniform(0,1), (l,l,l)])

	def gaussian(x, a, b, c, d=0):
		return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

	def pixel(x, width=100, map=[], spread=random.randint(1,3)):
		width = float(width)
		r = sum([gaussian(x, p[1][0], p[0] * width, width/(spread*len(map))) for p in map])
		g = sum([gaussian(x, p[1][1], p[0] * width, width/(spread*len(map))) for p in map])
		b = sum([gaussian(x, p[1][2], p[0] * width, width/(spread*len(map))) for p in map])
		return min(1.0, r), min(1.0, g), min(1.0, b)

	for x in range(im.size[0]):
		r, g, b = pixel(x, width=3000, map=heatmap)
		r, g, b = [int(256*v) for v in (r, g, b)]
		for y in range(im.size[1]):
			lighting[x, y] = r, g, b

	angle=random.randint(1,359)
	lighting=np.uint8(np.asarray(im))
	lighting = imutils.rotate(lighting, angle)

	edited_image = cv2.addWeighted(image, 0.5, lighting, 0.5, 0)

	return edited_image,mask

def speckling(image,mask):
	if np.amax(image)<=1.:
		image=(image*255).astype(int)

	for i in range(random.randint(50,1000)):
		image[random.randint(0,image.shape[0]-1),random.randint(0,image.shape[1]-1),:]=np.zeros(3)

	return image,mask

def randRange(a, b):
	'''
	a utility functio to generate random float values in desired range
	'''
	return np.random.rand() * (b - a) + a


def randomAffine(im,mask):
	'''
	wrapper of Affine transformation with random scale, rotation, shear and translation parameters
	'''
	tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
							rotation=randRange(-0.25, 0.25),
							shear=randRange(-0.2, 0.2),
							translation=(randRange(-im.shape[0]//10, im.shape[0]//10), 
										 randRange(-im.shape[1]//10, im.shape[1]//10)))
	im = warp(im, tform.inverse,mode='constant',cval=0,preserve_range=True)
	mask = warp(mask, tform.inverse,mode='constant',cval=0,order=0,preserve_range=True)

	return im, mask


def randomPerspective(im,mask):
	'''
	wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
	'''
	region = 1/8
	A = np.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
	B = np.array([[int(randRange(0, im.shape[1] * region)), int(randRange(0, im.shape[0] * region))], 
				  [int(randRange(0, im.shape[1] * region)), int(randRange(im.shape[0] * (1-region), im.shape[0]))], 
				  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(im.shape[0] * (1-region), im.shape[0]))], 
				  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(0, im.shape[0] * region))], 
				 ])

	pt = ProjectiveTransform()
	pt.estimate(A, B)

	im = warp(im, pt, output_shape=im.shape[:2],order=0,preserve_range=True)
	mask = warp(mask, pt, output_shape=im.shape[:2],order=0,preserve_range=True)

	return im, mask


def randomCrop(im,mask):
	'''
	croping the image in the center from a random margin from the borders
	'''
	margin = 1/10
	start = [int(randRange(0, im.shape[0] * margin)),
			 int(randRange(0, im.shape[1] * margin))]
	end = [int(randRange(im.shape[0] * (1-margin), im.shape[0])), 
		   int(randRange(im.shape[1] * (1-margin), im.shape[1]))]
	im = im[start[0]:end[0], start[1]:end[1]]
	mask = mask[start[0]:end[0], start[1]:end[1]]

	return im, mask


def randomIntensity(im,mask):
	'''
	rescales the intesity of the image to random interval of image intensity distribution
	'''
	im = rescale_intensity(im,
							 in_range=tuple(np.percentile(im, (randRange(0,10), randRange(90,100)))),
							 out_range=tuple(np.percentile(im, (randRange(0,10), randRange(90,100)))))

	return im, mask

def randomGamma(im,mask):
	'''
	Gamma filter for contrast adjustment with random gamma value.
	'''
	im = adjust_gamma(im, gamma=randRange(0.5, 1.5))

	return im, mask

def randomGaussian(im,mask):
	'''
	Gaussian filter for bluring the image with random variance.
	'''
	im = gaussian(im, sigma=randRange(0, 5),multichannel=False)

	return im, mask
	
def randomFilter(im,mask):
	'''
	randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
	filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
	'''
	Filters = [adjust_log, randomGamma, randomGaussian, equalize_adapthist, randomIntensity,equalize_hist]

	filt = random.choice(Filters)

	if filt in [randomGamma, randomGaussian, randomIntensity]:
		im,mask = filt(im,mask)
	else:
		im=filt(im)

	return im,mask


def randomNoise(im,mask):
	'''
	random gaussian noise with random variance.
	'''
	var = randRange(0.001, 0.01)
	im = random_noise(im, var=var)
	var = randRange(0.01, 0.3)
	im = random_noise(im, amount=var,mode='s&p')

	return im, mask
	
def generate_shadow_coordinates(imshape, no_of_shadows=1):    
	vertices_list=[]    
	for index in range(no_of_shadows):        
		vertex=[]        
		for dimensions in range(np.random.randint(3,15)): ## Dimensionality of the shadow polygon            
			vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))        
		vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices         
		vertices_list.append(vertices)    
	return vertices_list ## List of shadow vertices

def add_shadow(im,mask,no_of_shadows=1):    
	image = np.float32(im)
	no_of_shadows=random.randint(1,5)
	msk = np.zeros_like(image)     
	imshape = image.shape    
	vertices_list= generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices    
	for vertices in vertices_list:         
		cv2.fillPoly(msk, vertices, 255) ## adding all shadow polygons on empty msk, single 255 denotes only red channel        
	image[:,:,1][msk[:,:,0]==255] = image[:,:,1][msk[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered
	return image,mask

def augment(im,mask, Steps=[randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]):
	'''
	image augmentation by doing a sereis of transfomations on the image.
	'''

	for step in Steps:
		im,mask = step(im,mask)
		print(step.__name__)
		if np.amax(im)<=1.:
			image=(im*255).astype(int)
		else:
			image=im
		image=Image.fromarray(np.uint8(image))
		image.save('00_'+step.__name__+'.png','PNG')

	return im,mask

def augment_image(im, mask, target_size=None, bg_image_list=None,
	contrast_adjust=True, augment=True):

	dim = target_size

	if isinstance(im, (np.ndarray, np.generic)) == False:
			im = np.asarray(im)

	im = np.uint8(im[:,:,:3])

	if im.shape[0] != dim and im.shape[1] != dim:
		im = cv2.resize(im, (dim, dim), interpolation = cv2.INTER_AREA)

	if mask.shape[0] != dim and mask.shape[1] != dim:
		mask = cv2.resize(mask, (dim, dim), interpolation = cv2.INTER_NEAREST)

	if augment:
		if len(mask.shape) == 3:
			mask = np.argmax(mask, axis = -1)
		empty_mask = np.empty((mask.shape[0], mask.shape[1], 3))
		for i in range(3):
			empty_mask[:, :, i] = mask
		mask = empty_mask

	if augment:

		if bg_image_list != None:
			bg = Image.open(random.choice(bg_image_list)).convert('RGB') 
			random_ratio = random.uniform(1,1.5)
			bg_width = int(random_ratio * im.shape[0])
			bg_height = int(random_ratio * im.shape[1])
			
			bg = bg.resize((bg_height,bg_width))
			bg = np.array(bg) / 255
			mask_bg = np.zeros((bg_width,bg_height,3))

			bg[(bg.shape[0] - im.shape[0]) // 2 : ((bg.shape[0] - im.shape[0]) // 2) + im.shape[0],
				(bg.shape[1] - im.shape[1]) // 2 : ((bg.shape[1] - im.shape[1]) // 2) + im.shape[1],
				:] = im/255

			mask_bg[(bg.shape[0] - im.shape[0]) // 2 : ((bg.shape[0] - im.shape[0]) // 2) + im.shape[0],
				(bg.shape[1] - im.shape[1]) // 2 : ((bg.shape[1] - im.shape[1]) // 2) + im.shape[1],
				:] = mask

			im = bg
			mask = mask_bg

		if contrast_adjust:
			steps = [add_shadow, randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop,
				adjust_log, randomGamma, randomGaussian, equalize_adapthist, randomIntensity,equalize_hist,
				speckling,add_differential_lighting]
		else:
			steps = [add_shadow, randomAffine, randomPerspective]

		random.shuffle(steps)
		steps = steps[:random.randint(1,len(steps)-1)]

		for filt in steps:
			if filt in [adjust_log,equalize_adapthist,equalize_hist]:
				im = filt(im)
			else:
				try:
					im, mask = filt(im,mask)
				except:
					pass
			if np.amax(im) > 1.:
				im = im / 255

		if np.amax(im) <= 1.:
			im = (im * 255).astype(int)

		mask = indices_to_one_hot(mask[:, :, 0], n_classes, waves_only = config.waves_only)

	elif len(mask.shape) == 2:
			mask = indices_to_one_hot(mask, n_classes, waves_only = config.waves_only)

	if im.shape[0] != dim and im.shape[1] != dim:
		im = cv2.resize(np.uint8(im), (dim, dim), interpolation = cv2.INTER_AREA)

	if mask.shape[0] != dim and mask.shape[1] != dim:
		mask = cv2.resize(mask, (dim, dim), interpolation = cv2.INTER_NEAREST)

	return im, mask

if __name__ == '__main__':
	dim=1120
	n_classes=8

	im, mask=Training_Generator()

	im = np.asarray(im)

	X,mask = Augment_Image(im,mask,dim)
	print(X.shape,mask.shape)
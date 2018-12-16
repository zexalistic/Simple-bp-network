import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, color,transform

def tail(img):
	for i in range(img.shape[0]):
		if (np.mean(img[i,:]) != 1):
			top = i
			break
	for i in reversed(range(top,img.shape[0])):
		if (np.mean(img[i,:]) != 1):
			bottom = i
			break
	for i in range(img.shape[1]):
		if (np.mean(img[:,i]) != 1):
			left = i
			break
	for i in reversed(range(left,img.shape[1])):
		if (np.mean(img[:,i]) == 1):
			right = i
			break

	return top,bottom,left,right

def pool(img):
	h,w = img.shape
	a = h//10
	b = w //10
	size = a*b
	K = np.zeros((10,10))
	for i in range(10):
		for j in range(10):
			sum_ = 0
			for k in range(a):
				for l in range(b):
					sum_ += img[k+i*a][l+j*b]
			K[i][j] = sum_/size
	
	return K



path = os.getcwd() + '/Hnd/Img/Sample001/' + 'img001-012.png'
img = io.imread(path).astype(np.double)
img = color.rgb2gray(img)
img = img/255
img = img.astype(int)
top,bottom,left,right = tail(img)
width = right - left
height = bottom - top
new = np.zeros((height,width))
for i in range(height):
	for j in range(width):
		new[i][j] = img[top + i][left + j]
af = pool(new)
print(af.shape)
print(af)

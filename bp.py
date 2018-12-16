import os
import numpy as np
from skimage import io, color, transform
import random
import math

batch_size = 31
learning_rate = 0.001
input_dim = 100
hidden_dim = 100
output_dim = 62

def tail(img):
	top = 0
	bottom = img.shape[0]
	left = 0
	right = img.shape[1]
	for i in range(img.shape[0]):
		if (np.mean(img[i,:]) != 1):
			top = i
			break
	for i in reversed(range(top,img.shape[0])):
		if (np.mean(img[i,:]) == 1):
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

def preprocess(img):
	t,b,l,r = tail(img)
	width = r - l
	height = b - t
	new = np.zeros((height,width))
	for i in range(height):
		for j in range(width):
			new[i][j] = img[t + i][l + j]
	pre = pool(new)

	return pre

def import_data():
	train_x, train_y, test_x, test_y = [],[],[],[]

	path = os.getcwd() + '/Hnd/Img'
	folder_list = os.listdir(path)
	for fo in folder_list:
		folder_path = path + '/' + fo
		img_num = int(fo.split('e')[1])
		img_class = np.zeros(62)
		img_class[img_num - 1] = 1
		img_list = os.listdir(folder_path)
		test_set = random.sample(img_list,5)
		for test in test_set:
			img_list.remove(test)
		train_set = img_list
		#for debug
		train_set = random.sample(img_list,5)
		for img in test_set:
			img_path = folder_path + '/' + img
			img = io.imread(img_path).astype(np.double)
			img = color.rgb2gray(img)
			img = img/255
			img = img.astype(int)
			img = preprocess(img)
			test_x.append(img.reshape(-1))
			test_y.append(img_class.reshape(-1))
		for img in train_set:
			img_path = folder_path + '/' + img
			img = io.imread(img_path).astype(np.double)
			img = color.rgb2gray(img)
			img = img/255
			img = img.astype(int)
			img = preprocess(img)
			train_x.append(img.reshape(-1))
			train_y.append(img_class.reshape(-1))

	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)

	return train_x, train_y, test_x, test_y

def next_batch(iters,batch,train_x, train_y):
	train_size = train_x.shape[0]
	start = ((iters*batch) % train_size)
	if iters%20 == 0:
		idx = np.arange(train_size)
		np.random.shuffle(idx)
		train_x = train_x[idx]
		train_y = train_y[idx]
		
	end = start + batch
	x = train_x[start:end]
	y = train_y[start:end]

	return x,y,train_x,train_y

def normalize(x):
	m,n = x.shape
	K = np.zeros((m,n))
	#x_mean = np.mean(x, axis = 0)
	x_mean = np.mean(x)
	#x_std = np.std(x, axis = 0)
	x_std = np.std(x)

	for j in range(n):
		for i in range(m):
			K[i][j] = (x[i][j] - x_mean)/x_std

	return K

class BP_Network():
	def __init__(self):
		self.w1 = np.random.normal(0,0.01,input_dim*hidden_dim).reshape(input_dim, hidden_dim)
		self.w2 = np.random.normal(0,0.01,hidden_dim*output_dim).reshape(hidden_dim,output_dim)
		self.w3 = np.random.normal(0,0.01,output_dim*output_dim).reshape(output_dim,output_dim)
		self.b1 = np.ones(hidden_dim)
		self.b2 = np.ones(output_dim)
		self.b3 = np.ones(output_dim)
		self.output = None
		self.l1 = None
		self.l2 = None
		self.l3 = None

	def sigmoid(self,x):
		return 1.0/(1.0 + np.exp(-x))
	
	def derivative(self,x):
		return x*(1.0 - x)
	
	def forward_prop(self, input_data, label):
		self.l1 = input_data
		self.l2 = self.sigmoid( np.add(np.dot(self.l1, self.w1), self.b1) )
		self.l3 = self.sigmoid( np.add(np.dot(self.l2, self.w2), self.b2) )
		self.output = self.sigmoid( np.add(np.dot(self.l3, self.w3), self.b3) )
		err = label - self.output 
		
		return self.output, err
	
	def backward_prop(self, error):
		err = error
		l3_delta = err * self.derivative(self.output) * self.derivative(self.out)
		l2_delta = np.dot(l3_delta, self.w3.T) * self.derivative(self.l3)
		l1_delta = np.dot(l2_delta, self.w2.T) * self.derivative(self.l2)
		self.w1 += learning_rate * np.dot(self.l1.T, l1_delta)
		self.w2 += learning_rate * np.dot(self.l2.T, l2_delta)
		self.w3 += learning_rate * np.dot(self.l3.T, l3_delta)
		
if __name__ == '__main__':
	train_X, train_Y, test_x, test_y = import_data()

	bp = BP_Network()
	for iters in range(15000):
		input_batch, label_batch, train_X, train_Y = next_batch(iters,batch_size,train_X,train_Y)
		_, error = bp.forward_prop(input_batch, label_batch)
		bp.backward_prop(error)
		if iters % 1000 == 0:
			print('The loss is %f' %(np.mean(np.abs(error))))

	result, _ = bp.forward_prop(test_x,test_y)
	print(result)
	acc = np.equal(np.argmax(result,1), np.argmax(test_y,1))
	cnt = 0
	for i in range(acc.shape[0]):
		if acc[i] == True:
			print(i)
			cnt += 1
	accu = cnt/acc.shape[0]
	print('The accuracy is %f' %accu)

import os
import numpy as np
from skimage import io, color, transform
import random
import math
import tensorflow as tf

batch_size = 31
learning_rate = 0.001
input_dim = 108	
hidden_dim = 100
output_dim = 62

def PCA(x,k):
	x -= np.mean(x,axis=0)
	cov = np.cov(x.T)
	evals, evecs = np.linalg.eigh(cov)
	idx = np.argsort(evals)[::-1][:k]
	evecs = evecs[:,idx]

	return np.dot(evecs.T, x.T).T

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
#			img = transform.resize(img,[90,120])
			img = transform.resize(img,[9,12])
			#img = PCA(img,20)
			test_x.append(img.reshape(-1))
			test_y.append(img_class.reshape(-1))
		for img in train_set:
			img_path = folder_path + '/' + img
			img = io.imread(img_path).astype(np.double)
			img = color.rgb2gray(img)
#			img = transform.resize(img,[90,120])
			img = transform.resize(img,[9,12])
			#img = PCA(img,20)
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
	if start == 0:
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
	x_mean = np.mean(x, axis = 0)
	x_std = np.std(x, axis = 0)
	#x_max = np.amax(x, axis = 0)
	#x_min = np.amin(x, axis = 0)

	for j in range(n):
		for i in range(m):
			K[i][j] = (x[i][j] - x_mean[j])/x_std[j]
#			K[i][j] = ((x_max[j] - x[i][j])/(x_max[j] - x_min[j]))

	return K

def glorot_init(shape):
	return tf.random_normal(shape = shape, stddev = 1./tf.sqrt(shape[0]/2.))

weights = {
	'input': tf.Variable(glorot_init([input_dim, hidden_dim])),
	'hidden': tf.Variable(glorot_init([hidden_dim, output_dim])),
	'output': tf.Variable(glorot_init([output_dim, output_dim])),
}
biases = {
	'input': tf.Variable(tf.ones([hidden_dim])),
	'hidden': tf.Variable(tf.ones([output_dim])),
	'output': tf.Variable(tf.ones([output_dim])),			
}

#Feature extractor
def feature_extractor(x):
	input_layer = tf.matmul(x, weights['input'])
	input_layer = tf.add(input_layer, biases['input'])
	input_layer = tf.nn.sigmoid(input_layer)

	hidden_layer = tf.matmul(input_layer, weights['hidden'])
	hidden_layer = tf.add(hidden_layer, biases['hidden'])
	hidden_layer = tf.nn.sigmoid(hidden_layer)

	output_layer = tf.matmul(hidden_layer, weights['output'])
	output_layer = tf.add(output_layer, biases['output'])
	output_layer = tf.nn.sigmoid(output_layer)

	return output_layer

# Build Networks
# Network Inputs
img_input = tf.placeholder(tf.float32, shape=[None, input_dim], name='image_input')
label = tf.placeholder(tf.float32, shape =[None, output_dim], name = 'Label')

feature_output = feature_extractor(img_input)

loss =  tf.nn.l2_loss(feature_output - label)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gen_vars = [weights['input'], weights['hidden'], weights['output']]

# Create training operations
train = optimizer.minimize(loss, var_list = gen_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start 
with tf.Session() as sess:
	sess.run(init)
	train_X, train_Y, test_x, test_y = import_data()
	train_X = normalize(train_X)
	test_x = normalize(test_x)

	for epoch in range(1,15001):
		input_batch, label_batch, train_X, train_Y = next_batch(epoch,batch_size,train_X,train_Y)
		_,dl = sess.run([train, loss],feed_dict={img_input: input_batch, label: label_batch})
		if epoch % 1000 == 0:
			print('Step %i: Generator Loss: %f' % (epoch, dl))

	result = sess.run([feature_output], feed_dict = {img_input: test_x})
	result = np.asarray(result).reshape(310,62)
	acc = np.equal(np.argmax(result,1), np.argmax(test_y,1))
	cnt = 0
	for i in range(acc.shape[0]):
		if acc[i] == True:
			cnt += 1
	accu = cnt/acc.shape[0]
	print('The accuracy is %f' %accu)

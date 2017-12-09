from __future__ import division

### git@yash0307 ###

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, merge, Activation, Conv1D, Input, MaxPooling1D, Convolution1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import json
import sys
import random
import scipy.io as sio
from PIL import Image

def initialize_net(train_params):
	model = Sequential()
	model.add(Convolution1D(nb_filter=5, 
				filter_length=10, 
				init='glorot_uniform', 
				border_mode='same', 
				input_shape=(None, 3), 
				bias=True))
	model.add(Activation('tanh'))
	model.add(Convolution1D(nb_filter=10,
				filter_length=20, 
				init='glorot_uniform', 
				border_mode='same', 
				bias=True))
	model.add(Activation('tanh'))
	model.add(Convolution1D(nb_filter=20,
				filter_length=20,
				init='glorot_uniform',
				border_mode='same',
				bias=True))
	model.add(GlobalAveragePooling1D(input_shape=model.output_shape[1:]))
	model.add(Activation('tanh'))
	model.add(Dense(input_dim=20, 
			output_dim=2,
			init='glorot_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Dense(input_dim=2, 
			output_dim=1, 
			init='glorot_uniform'))
	model.add(Activation('softmax'))
	return model

def initialize_params(train_data, data):

	train_params = {'batch_size':256, 
			'max_size':256, 
			'base_lr':0.0001, 
			'decay_steps':3,
			'decay_factor':0.5, 
			'num_epochs':12, 
			'neg_samples':len(data[0]), 
			'pos_samples':len(data[1]), 
			'total_samples':len(data[0])+len(data[1]), 
			'checkpoint':5}

	return train_params

def get_train_data(train_data, train_labels):
	data = {1:[], 0:[]}
	num_images = train_data.shape[1]
	for i in range(0, num_images):
		given_image_sp = train_data[0][i]
		given_image_lb = train_labels[i][0]
		num_sp = given_image_lb.shape[1]
		for j in range(0, num_sp):
			given_label = given_image_lb[0][j]
			if given_label == 0:
				given_rep = np.asarray(given_image_sp[j][:], dtype='float')
				data[0].append(given_rep)
			elif given_label == 1:
				given_rep = np.asarray(given_image_sp[j][:], dtype='float')
				data[1].append(given_rep)
			else:
				print('SOMETHING IS WRONG !')
	return data

def load_data(data, train_params):
	data_frac = 0.5
	X_temp = np.zeros((train_params['batch_size'], train_params['max_size'], 3))
	Y_temp = np.zeros((train_params['batch_size'], 1))
	idx = random.sample(range(0,train_params['pos_samples']), int(train_params['batch_size']*data_frac+2))
	for i in range(0, int(train_params['batch_size']*data_frac)):
		Y_temp[i] = float(1)
		sam = data[1][idx[i]]
		sam_len = sam.shape[0]
		X_temp[i, :sam_len, :] = sam
	idx = random.sample(range(0, train_params['neg_samples']), int(train_params['batch_size']-(train_params['batch_size']*data_frac)+2))
	for i in range(int(train_params['batch_size']*data_frac), train_params['batch_size']):
		Y_temp[i] = float(0)
		sam = data[0][idx[i-int(train_params['batch_size']*data_frac)]]
		sam_len = sam.shape[0]
		X_temp[i, :sam_len, :] = sam
        X = np.zeros((train_params['batch_size'], train_params['max_size'], 3))
        Y = np.zeros((train_params['batch_size'], 1))
	perm_idx = np.random.permutation(train_params['batch_size'])
	for i in range(0, train_params['batch_size']):
		X[i,:,:] = X_temp[perm_idx[i],:,:]
		Y[i,:] = Y_temp[perm_idx[i],:]
	return (X,Y)

if __name__ == '__main__':

	f_out = open('results.txt','w')
	train_data = sio.loadmat('../all_Q.mat')['all_Q']
	train_labels = sio.loadmat('../all_superpixel_labels.mat')['all_superpixel_labels']
	data = get_train_data(train_data, train_labels)
	train_params = initialize_params(train_data, data)

	model = initialize_net(train_params)
	model.summary()

	model.compile(loss='mean_squared_error',
		optimizer=optimizers.SGD(lr=train_params['base_lr'], momentum=0.9),
		metrics=['accuracy'])

	train_datagen = ImageDataGenerator(
				featurewise_center=True,
				featurewise_std_normalization=True)

	for epoch in range(0, train_params['num_epochs']):
		num_iterations = int(train_params['total_samples']/train_params['batch_size']) + 1
		for iteration in range(0, num_iterations):
			print 'Epoch : ' + str(epoch) + ' | Iteration : ' + str(iteration)
			given_data = load_data(data, train_params)
			X = given_data[0]
			Y = given_data[1]
			model.fit(X,Y,
				epochs=1,
				verbose=1)
		if epoch%train_params['decay_steps'] == 0 and epoch != 0:
			print ' Changing learning rate ... '
			lr = K.get_value(model.optimizer.lr)
			K.set_value(model.optimizer.lr, lr*train_params['decay_factor'])
			print("lr changed to {}".format(lr*train_params['decay_factor']))
		if epoch%train_params['checkpoint'] == 0 and epoch != 0:
			print ' Saving model ... '
			model_name = 'model_' + str(epoch) + '.h5'
			model.save(model_name)
		if epoch%1 == 0:
			acu_pos = 0
			acu_neg = 0
			for i in range(0, int(train_params['pos_samples']/train_params['batch_size'])):
				X = np.zeros((train_params['batch_size'], train_params['max_size'], 3))
				Y = np.zeros((train_params['batch_size'], 1))
				for j in range(0, train_params['batch_size']):
					sam = data[1][i*train_params['batch_size'] + j]
					sam_len = sam.shape[0]
					X[j, :sam_len, :] = sam
					Y[j] = float(1)
				pred = model.evaluate(X,Y, batch_size=train_params['batch_size'])
				print(pred)
				acu_pos = acu_pos + pred[1]
			for i in range(0, int(train_params['neg_samples']/train_params['batch_size'])):
				X = np.zeros((train_params['batch_size'], train_params['max_size'], 3))
				Y = np.zeros((train_params['batch_size'], 1))
				for j in range(0, train_params['batch_size']):
					sam = data[0][i*train_params['batch_size'] + j]
					sam_len = sam.shape[0]
					X[j, :sam_len, :] = sam
					Y[j] = float(0)
				pred = model.evaluate(X,Y, batch_size=train_params['batch_size'])
				print(pred)
				acu_neg = acu_neg + pred[1]
			acu_pos = float(acu_pos)/float(int(train_params['pos_samples']/train_params['batch_size'])) 
			acu_neg = float(acu_neg)/float(int(train_params['neg_samples']/train_params['batch_size']))
			f_out.write('acu_pos: ' + str(acu_pos)+', acu_neg: '+str(acu_neg)+'\n')

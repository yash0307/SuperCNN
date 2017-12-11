from __future__ import division

### git@yash0307 ###

import numpy as np
import json
import sys
import random
import scipy.io as sio
from PIL import Image
import sklearn
from sklearn import svm

def initialize_params(train_data, data):

	train_params = {
			'max_size':256, 
			'neg_samples':len(data[0]), 
			'pos_samples':len(data[1]), 
			'total_samples':len(data[0])+len(data[1])}

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
				given_rep = np.zeros((256,3))
				sam = np.asarray(given_image_sp[j][:], dtype='float')
				sam_len = sam.shape[0]
				given_rep[:sam_len, :] = sam
				data[0].append(given_rep)
			elif given_label == 1:
				given_rep = np.zeros((256, 3))
				sam = np.asarray(given_image_sp[j][:], dtype='float')
				sam_len = sam.shape[0]
				given_rep[:sam_len, :] = sam
				data[1].append(given_rep)
			else:
				print('SOMETHING IS WRONG !')
	return data

if __name__ == '__main__':

	classes = ['fg', 'bg']
	f_out = open('results.txt','w')
	train_data = sio.loadmat('../all_Q.mat')['all_Q']
	train_labels = sio.loadmat('../all_superpixel_labels.mat')['all_superpixel_labels']
	data = get_train_data(train_data, train_labels)
	train_params = initialize_params(train_data, data)
	
	X = np.zeros((train_params['total_samples'], train_params['max_size'], 3))
	Y = np.zeros((train_params['total_samples'], 1))

	for i in range(0,train_params['pos_samples']):
		X[i,:,:] = data[1][i]
		Y[i] = 1
	for i in range(train_params['pos_samples'], train_params['total_samples']):
		X[i,:,:] = data[0][i-train_params['pos_samples']]
		Y[i] = 0
	clf = svm.SVC(kernel='rbf')
	clf.fit(X,Y)

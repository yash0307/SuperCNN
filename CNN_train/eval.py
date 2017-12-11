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

def initialize_params(train_data, data):

        train_params = {'batch_size':256,
                        'max_size':256,
                        'base_lr':0.001,
                        'decay_steps':5,
                        'decay_factor':0.5,
                        'num_epochs':15,
                        'neg_samples':len(data[0]),
                        'pos_samples':len(data[1]),
                        'total_samples':len(data[0])+len(data[1]),
                        'checkpoint':1}

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
				sys.exit(1)
        return data

if __name__ == '__main__':

        f_out = open('results.txt','w')
        train_data = sio.loadmat('../all_Q.mat')['all_Q']
        train_labels = sio.loadmat('../all_superpixel_labels.mat')['all_superpixel_labels']
        data = get_train_data(train_data, train_labels)
        train_params = initialize_params(train_data, data)

	model = load_model('model_14.h5')
	num_images = train_data.shape[1]
	avg_acu = 0
	out_mat = np.zeros((num_images, train_params['max_size']))
	for i in range(0, num_images):
		given_image_sp = train_data[0][i]
		given_image_lb = train_labels[i][0]
		num_sp = given_image_lb.shape[1]
		acu = 0
		for j in range(0, num_sp):
			given_label = given_image_lb[0][j]
			X = np.zeros((1,train_params['max_size'], 3))
			if given_label == 0:
				given_rep = np.asarray(given_image_sp[j][:], dtype='float')
				sam_len = given_rep.shape[0]
				X[0,:sam_len, :] = np.true_divide(given_rep, given_rep.max())
				pred = model.predict(X)
				pred_idx = np.where(pred == pred.max())[1][0]
				if (pred.max() < 0.60) and (pred_idx == 1):
					pred_idx = 0
				out_mat[i][j] = pred_idx
				if pred_idx == given_label:
					acu += 1
				else:
					pass
			elif given_label == 1:
				given_rep = np.asarray(given_image_sp[j][:], dtype='float')
				sam_len = given_rep.shape[0]
				X[0,:sam_len, :] = np.true_divide(given_rep, given_rep.max())
				pred = model.predict(X)
				pred_idx = np.where(pred == pred.max())[1][0]
				out_mat[i][j] = pred_idx
				if pred_idx == given_label:
					acu += 1
				else:
					pass
			else:
				print('SOMETHING IS WRONG !')
				sys.exit(1)
		acu = float(acu)/float(num_sp)
		print('Given Image Acu: ' + str(acu))
		avg_acu = avg_acu + acu
	avg_acu = float(avg_acu)/float(num_images)
	print('Over Acu: ' + str(avg_acu))
	sio.savemat('./out.mat', mdict={'out_mat':out_mat})

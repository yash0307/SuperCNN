from __future__ import division

### git@yash0307 ###

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, merge, Activation, Conv1D, Input, MaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import json
import sys
import random
from PIL import Image

def initialize_net():
	model = Sequential()
	model.add(Input(shape=(None, None, 3)))
	model.add(Conv1D(filters=5, kernal_size=4, strides=1, padding='valid', activation='relu'))
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
	model.add(Conv1D(filters=5, kernal_size=2, strides=1, padding='valid', activation='relu'))
	model.add(GlobalAveragePooling1D)
	return model

model = initialize_net()
model.summary()

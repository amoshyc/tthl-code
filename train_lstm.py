import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import *
from keras.optimizers import *

from myutils import get_callbacks


model = Sequential()
model.add(BatchNormalization(input_shape=(5, 224, 224, 3)))
model.add(TimeDistributed(Conv2D(5, kernel_size=5, strides=2)))
model.add(TimeDistributed(Conv2D(10, kernel_size=5, strides=2)))
model.add(TimeDistributed(Conv2D(15, kernel_size=4, strides=1)))
model.add(BatchNormalization())
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=3)))
model.add(TimeDistributed(Conv2D(20, kernel_size=4, strides=1)))
model.add(TimeDistributed(Conv2D(15, kernel_size=3, strides=1)))
model.add(TimeDistributed(Conv2D(10, kernel_size=3, strides=1)))
model.add(BatchNormalization())
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=3)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model_arg = {
    'loss': 'binary_crossentropy',
    'optimizer': 'sgd',
    'metrics': ['binary_accuracy']
}
model.compile(**model_arg)
model.summary()

train = np.load('npz/win_train.npz')
x_train, y_train = train['xs'], train['ys']
val = np.load('npz/win_val.npz')
x_val, y_val = val['xs'], val['ys']

fit_arg = {
    'x': x_train,
    'y': y_train,
    'batch_size': 40,
    'epochs': 50,
    'shuffle': True,
    'validation_data': (x_val, y_val),
    'callbacks': get_callbacks('lstm'),
}
model.fit(**fit_arg)

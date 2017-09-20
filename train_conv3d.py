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

from data import *
from myutils import get_callbacks


def main():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(4, 224, 224, 3)))
    model.add(TimeDistributed(Conv2D(5, kernel_size=5, strides=2, activation='relu')))
    model.add(TimeDistributed(Conv2D(10, kernel_size=4, strides=2, activation='relu')))
    model.add(TimeDistributed(Conv2D(15, kernel_size=3, strides=1, activation='relu')))
    model.add(BatchNormalization())
    model.add(TimeDistributed(MaxPooling2D(pool_size=3)))
    model.add(Conv3D(4, kernel_size=2, strides=1, activation='relu'))
    model.add(Conv3D(1, kernel_size=2, strides=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
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

    train = np.load('d2/window_train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('d2/window_val.npz')
    x_val, y_val = val['xs'], val['ys']

    fit_arg = {
        'x': x_train,
        'y': y_train,
        'batch_size': 250,
        'epochs': 100,
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': get_callbacks('conv3d'),
    }
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()

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
from utils import get_callbacks


def main():
    with tf.device('/gpu:3'):
        model = Sequential()
        model.add(TimeDistributed(BatchNormalization(), input_shape=(TIMESTEPS, 224, 224, 3)))
        model.add(TimeDistributed(Conv2D(4, kernel_size=5, strides=3, activation='relu')))
        model.add(TimeDistributed(Conv2D(8, kernel_size=5, strides=2, activation='relu')))
        model.add(TimeDistributed(Conv2D(12, kernel_size=3, strides=1, activation='relu')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D(pool_size=3)))
        model.add(Conv3D(4, kernel_size=5, strides=1, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

    model_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['binary_accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    n_train, n_val = 5000, 1000
    x_train = np.zeros((n_train, TIMESTEPS, 224, 224, 3), dtype=np.float32)
    y_train = np.zeros((n_train, 1), dtype=np.uint8)
    x_val = np.zeros((n_val, TIMESTEPS, 224, 224, 3), dtype=np.float32)
    y_val = np.zeros((n_val, 1), dtype=np.uint8)

    train_gen = window_generator(WINDOW_TRAIN, 1)
    val_gen = window_generator(WINDOW_VAL, 1)

    print('Loading data...', end='')
    for i in range(n_train):
        x, y = next(train_gen)
        x_train[i] = x
        y_train[i] = y
    for i in range(n_val):
        x, y = next(val_gen)
        x_val[i] = x
        y_val[i] = y
    print('ok')

    fit_arg = {
        'x': x_train,
        'y': y_train,
        'batch_size': WINDOW_BATCH_SIZE,
        'epochs': 30,
        'validation_data': (x_val, y_val),
        'shuffle': True
    }

    model.fit(**fit_arg)


if __name__ == '__main__':
    main()
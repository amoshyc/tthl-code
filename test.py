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

from tqdm import tqdm

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

    n_train, n_val = 4000, 800
    x_train = np.zeros((n_train, TIMESTEPS, 224, 224, 3), dtype=np.float32)
    y_train = np.zeros((n_train, 1), dtype=np.uint8)
    x_val = np.zeros((n_val, TIMESTEPS, 224, 224, 3), dtype=np.float32)
    y_val = np.zeros((n_val, 1), dtype=np.uint8)

    x_train_npys = sorted(WINDOW_TRAIN.glob('x_*.npy'))[:n_train // 200]
    y_train_npys = sorted(WINDOW_TRAIN.glob('y_*.npy'))[:n_train // 200]
    x_val_npys = sorted(WINDOW_VAL.glob('x_*.npy'))[:n_val // 200]
    y_val_npys = sorted(WINDOW_VAL.glob('y_*.npy'))[:n_val // 200]
    for i, (x_npy, y_npy) in enumerate(tqdm(zip(x_train_npys, y_train_npys))):
        x_train[i * 200: i * 200 + 200] = np.load(x_npy)
        y_train[i * 200: i * 200 + 200] = np.load(y_npy)
    for i, (x_npy, y_npy) in enumerate(tqdm(zip(x_val_npys, y_val_npys))):
        x_val[i * 200: i * 200 + 200] = np.load(x_npy)
        y_val[i * 200: i * 200 + 200] = np.load(y_npy)

    
    

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
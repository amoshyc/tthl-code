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
from keras.applications.vgg16 import VGG16

from myutils import get_callbacks


def main():
    inp = Input(shape=(224, 224, 3))
    p1_inp = Input(shape=(100, 50, 3))
    p2_inp = Input(shape=(100, 50, 3))

    p_model = Sequential()
    p_model.add(BatchNormalization(input_shape=(100, 50, 3)))
    p_model.add(Conv2D(5, 5, activation='relu'))
    p_model.add(Conv2D(10, 4, activation='relu'))
    p_model.add(Conv2D(15, 4, activation='relu'))
    p_model.add(MaxPool2D(pool_size=3))
    p_model.add(Conv2D(15, 3, activation='relu'))
    p_model.add(Conv2D(8, 3, activation='relu'))
    p_model.add(Conv2D(4, 3, activation='relu'))
    p_model.add(MaxPool2D(pool_size=2))
    p_model.add(Flatten())

    x = BatchNormalization()(inp)
    x = VGG16(weights='imagenet', include_top=False, pooling='max')(x)
    p1 = p_model(p1_inp)
    p2 = p_model(p2_inp)

    fuse = concatenate([x, p1, p2])
    fuse = Dense(64, activation='relu')(fuse)
    fuse = Dropout(0.5)(fuse)
    fuse = Dense(16, activation='relu')(fuse)
    fuse = Dropout(0.5)(fuse)
    fuse = Dense(1, activation='sigmoid')(fuse)

    model = Model(inputs=[inp, p1_inp, p2_inp], outputs=fuse)

    model_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['binary_accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    train = np.load('npz/train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('npz/val.npz')
    x_val, y_val = val['xs'], val['ys']

    players_train = np.load('npz/players_train.npz')
    p1_train, p2_train = players_train['p1'], players_train['p2']
    players_val = np.load('npz/players_val.npz')
    p1_val, p2_val = players_val['p1'], players_val['p2']

    fit_arg = {
        'x': [x_train, p1_train, p2_train],
        'y': y_train,
        'batch_size': 30,
        'epochs': 100,
        'shuffle': True,
        'validation_data': ([x_val, p1_val, p2_val], y_val),
        'callbacks': get_callbacks('hltf'),
    }
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()

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
    base = VGG16(weights='imagenet', include_top=False, pooling='max')
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    for layer in base.layers:
        model.add(layer)
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model_arg = {
        'loss': 'categorical_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    train = np.load('npz/train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('npz/val.npz')
    x_val, y_val = val['xs'], val['ys']

    from keras.utils import to_categorical
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)

    fit_arg = {
        'x': x_train,
        'y': y_train,
        'batch_size': 40,
        'epochs': 50,
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': get_callbacks('gc'),
    }
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()

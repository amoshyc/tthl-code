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

from data import image_generator
from utils import get_callbacks


def main():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(4, kernel_size=5, strides=3, activation='relu'))
    model.add(Conv2D(8, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(12, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(30))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['binary_accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    n_train = 25000
    n_val = 5000
    batch_size = 40

    dataset = Path('~/dataset/').expanduser().resolve()
    video_dirs = sorted([x for x in dataset.iterdir() if x.is_dir()])
    train_dirs = [(dataset / 'video01/')]
    val_dirs = [(dataset / 'video00')]
    train_gen = image_generator(train_dirs, batch_size)
    val_gen = image_generator(val_dirs, batch_size)

    fit_arg = {
        'generator': train_gen,
        'steps_per_epoch': n_train // batch_size,
        'epochs': 30,
        'validation_data': val_gen,
        'validation_steps': n_val // batch_size,
        'callbacks': get_callbacks('cnn1bn')
    }

    model.fit_generator(**fit_arg)


if __name__ == '__main__':
    main()
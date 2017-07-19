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

    fit_arg = {
        'generator': image_train_gen,
        'steps_per_epoch': N_IMAGE_TRAIN // IMAGE_BATCH_SIZE,
        'epochs': 30,
        'validation_data': image_val_gen,
        'validation_steps': N_IMAGE_VAL // IMAGE_BATCH_SIZE,
        'callbacks': get_callbacks('cnn1bn')
    }

    model.fit_generator(**fit_arg)


if __name__ == '__main__':
    main()
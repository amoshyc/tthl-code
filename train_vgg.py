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

from utils import get_callbacks


def main():
    with tf.device('/gpu:2'):
        vgg = VGG16(weights='imagenet', include_top=False, pooling='max')
        x = vgg.output
        x = Dense(16)(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=vgg.input, outputs=x)

    model_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['binary_accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    train = np.load('npz/image_train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('npz/image_val.npz')
    x_val, y_val = val['xs'], val['ys']

    fit_arg = {
        'x': x_train, 
        'y': y_train,
        'batch_size': 50,
        'epochs': 100,
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': get_callbacks('vgg'),
    }
    model.fit(**fit_arg)

    # fit_gen_arg = {
    #     'generator': image_train_gen,
    #     'steps_per_epoch': N_IMAGE_TRAIN // IMAGE_BATCH_SIZE,
    #     'epochs': 30,
    #     'validation_data': image_val_gen,
    #     'validation_steps': N_IMAGE_VAL // IMAGE_BATCH_SIZE,
    #     'callbacks': get_callbacks('cnn')
    # }

    # model.fit_generator(**fit_gen_arg)


if __name__ == '__main__':
    main()
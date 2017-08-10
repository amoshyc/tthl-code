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
from keras.applications.mobilenet import MobileNet

from utils import get_callbacks


def main():
    inp = Input(shape=(224, 224, 3))
    x = BatchNormalization()(inp)
    x = MobileNet(
        weights='imagenet',
        include_top=False,
        pooling='max',
        input_shape=(224, 224, 3))(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)

    model_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adadelta',
        'metrics': ['binary_accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    train = np.load('d1/image_train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('d1/image_val.npz')
    x_val, y_val = val['xs'], val['ys']

    fit_arg = {
        'x': x_train,
        'y': y_train,
        'batch_size': 40,
        'epochs': 100,
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': get_callbacks('mobile'),
    }
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()
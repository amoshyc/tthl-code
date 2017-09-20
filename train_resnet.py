import json
from datetime import datetime

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
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, CSVLogger

from myutils import get_callbacks


inp = Input(shape=(224, 224, 3))
x = BatchNormalization()(inp)
x = ResNet50(weights='imagenet', include_top=False, pooling='max')(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=x)

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

fit_arg = {
    'x': x_train,
    'y': y_train,
    'batch_size': 40,
    'epochs': 50,
    'shuffle': True,
    'validation_data': (x_val, y_val),
    'callbacks': get_callbacks('resnet'),
}
model.fit(**fit_arg)

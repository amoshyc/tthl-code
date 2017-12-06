import argparse

import numpy as np
import scipy

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.layers import *
from keras.optimizers import *
from tqdm import tqdm

from myutils import get_callbacks

parser = argparse.ArgumentParser()
parser.add_argument('models', help='models', type=str, nargs='+')
args = parser.parse_args()

models = []
for i, m in enumerate(tqdm(args.models, ascii=True)):
    model = load_model(m, compile=False)
    model.name = f'model_{i}'
    model.trainable = False
    models.append(model)

inp = Input(shape=(224, 224, 3))
fuse = concatenate([m(inp) for m in models])
fuse = Dense(64, activation='relu')(fuse)
fuse = Dropout(0.5)(fuse)
fuse = Dense(16, activation='relu')(fuse)
fuse = Dropout(0.5)(fuse)
fuse = Dense(1, activation='sigmoid')(fuse)
model = Model(inputs=inp, outputs=fuse)

model_arg = {
    'loss': 'binary_crossentropy',
    'optimizer': SGD(lr=0.001, momentum=0.9),
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
    'batch_size': 30,
    'epochs': 20,
    'shuffle': True,
    'validation_data': (x_val, y_val),
    'callbacks': get_callbacks('ens2'),
}
model.fit(**fit_arg)

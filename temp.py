import json
import random
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
from tqdm import tqdm
from PIL import Image

def from_dir(directory, target_size=(224, 224)):
    path = Path(directory)

    imgs = sorted(list(path.glob('*/*.jpg')))
    random.shuffle(imgs)
    classes = list([x for x in path.iterdir() if x.is_dir()])
    classes.sort()

    n_samples = len(imgs)
    n_classes = len(classes)

    xs = np.zeros((n_samples, *target_size, 3), dtype=np.float32)
    ys = np.zeros((n_samples, ), dtype=np.uint8)
    for i, img_path in enumerate(tqdm(imgs, ascii=True)):
        img = Image.open(img_path)
        img = img.resize(target_size)
        xs[i] = np.array(img)
        ys[i] = classes.index(img_path.parent)

    return xs, ys

def main():
    inp = Input(shape=(224, 224, 3))
    x = BatchNormalization()(inp)
    x = VGG16(weights='imagenet', include_top=False, pooling='max')(x)
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

    x_train, y_train = from_dir('../tt-code/train/')
    x_val, y_val = from_dir('../tt-code/val')

    fit_arg = {
        'x': x_train,
        'y': y_train,
        'batch_size': 40,
        'epochs': 100,
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': get_callbacks('temp'),
    }
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()

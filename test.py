import json
import random
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def sample_windows(video_dirs, n_samples, timesteps, fps):
    windows = []
    for video_dir in video_dirs:
        video = VideoFileClip(str(video_dir / 'video.mp4'))
        n_frames = int(video.duration) * fps
        info = json.load((video_dir / 'info.json').open())

        label = np.zeros(n_frames, dtype=np.uint8)
        for s, e in zip(info['starts'], info['ends']):
            fs = round(s * fps)
            fe = round(e * fps)
            label[fs:fe] = 1

        cur = [(video, t - timesteps, t, label[t - 1])
               for t in range(timesteps, n_frames)]
        windows.extend(cur)

    return random.sample(windows, k=n_samples)


def window_gen(video_dirs, n_samples, batch_size, timesteps, fps):
    windows = sample_windows(video_dirs, n_samples, timesteps, fps)

    idx = 0
    x_batch = np.zeros((batch_size, timesteps, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    for (video, s, e, y) in windows:
        for i in range(e - s):
            img = video.get_frame((s + i) / fps)
            x_batch[idx][i] = scipy.misc.imresize(img, (224, 224))
        y_batch[idx] = y

        if idx + 1 == batch_size:
            yield x_batch, y_batch
        idx = (idx + 1) % batch_size


def window_data(video_dirs, n_samples, timesteps, fps):
    windows = sample_windows(video_dirs, n_samples, timesteps, fps)

    x_all = np.zeros((n_samples, timesteps, 224, 224, 3), dtype=np.float32)
    y_all = np.zeros((n_samples, 1), dtype=np.uint8)

    for idx, (video, s, e, y) in enumerate(tqdm(windows)):
        for i in range(e - s):
            img = video.get_frame((s + i) / fps)
            x_all[idx][i] = scipy.misc.imresize(img, (224, 224))
        y_all[idx] = y

    return x_all, y_all


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

from tt import WindowNpyGenerator
from utils import get_callbacks


def main():
    # with tf.device('/gpu:3'):
    model = Sequential()
    model.add(TimeDistributed(BatchNormalization(), input_shape=(4, 224, 224, 3)))
    model.add(TimeDistributed(Conv2D(4, kernel_size=5, strides=3, activation='relu')))
    model.add(TimeDistributed(Conv2D(8, kernel_size=5, strides=2, activation='relu')))
    model.add(TimeDistributed(Conv2D(12, kernel_size=3, strides=1, activation='relu')))
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


    dataset = Path('~/tthl-dataset/').expanduser()
    video_dirs = sorted(dataset.glob('video*/'))

    gen = WindowNpyGenerator(n_train=10000, n_val=2000, fps=2, timesteps=4, overlap=3)
    # gen.fit(video_dirs)

    fit_gen_arg = {
        'generator': gen.flow('train', 80),
        'steps_per_epoch': 10000 // 80,
        'epochs': 30,
        'validation_data': gen.flow('val', 80),
        'validation_steps': 2000 // 80,
        'callbacks': get_callbacks('conv3d')
    }
    model.fit_generator(**fit_gen_arg)

    # x_train, y_train = window_data(video_dirs[:3], 5000, 10, 5)
    # x_val, y_val = window_data(video_dirs[-1:], 1000, 10, 5)
    # fit_arg = {
    #     'x': x_train,
    #     'y': y_train,
    #     'batch_size': 80,
    #     'epochs': 30,
    #     'validation_data': (x_val, y_val)
    # }
    # model.fit(**fit_arg)


if __name__ == '__main__':
    main()

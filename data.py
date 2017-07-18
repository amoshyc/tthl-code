import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import *


def gen_image_npy(video_dirs, target_dir, n_samples):
    x_all = []
    y_all = []
    for video_dir in video_dirs:
        imgs = sorted((video_dir / 'frames/').iterdir())
        label = json.load((video_dir / 'label.json').open())['label']
        x_all.extend(imgs)
        y_all.extend(label)

    x_use, y_use = sample(x_all, y_all, k=n_samples)

    parts = split(x_use, y_use, k=1000)
    for idx, x_part, y_part in tqdm(parts):
        n = len(x_part)
        xs = np.zeros((n, 224, 224, 3), dtype=np.float32)
        ys = np.zeros((n, 1), dtype=np.uint8)
        for i in range(n):
            xs[i] = read_img(x_part[i])
            ys[i] = y_part[i]
        np.save(str(target_dir / 'x_{:05d}.npy'.format(idx)), xs)
        np.save(str(target_dir / 'y_{:05d}.npy'.format(idx)), ys)
        del xs, ys


def gen_window_npy(video_dirs, target_dir, n_samples, timesteps):
    x_all = []
    y_all = []
    for video_dir in video_dirs:
        n_frames = len(list((video_dir / 'frames/').iterdir()))
        labels = read_json(video_dir / 'label.json')['label']
        windows = [(video_dir, i, i + timesteps)
                   for i in range(n_frames - timesteps + 1)]
        x_all.extend(windows)
        y_all.extend([labels[e - 1] for (_, s, e) in windows])

    x_use, y_use = sample(x_all, y_all, k=n_samples)

    parts = split(x_use, y_use, k=200)
    for idx, x_part, y_part in tqdm(parts):
        n = len(x_part)
        xs = np.zeros((n, timesteps, 224, 224, 3), dtype=np.float32)
        ys = np.zeros((n, 1), dtype=np.uint8)
        for i in range(n):
            (video_dir, s, e) = x_part[i]
            for f in range(s, e):
                path = video_dir / 'frames' / '{:08d}.jpg'.format(f)
                xs[i][f - s] = read_img(path)
            ys[i] = y_part[i]

        np.save(str(target_dir / 'x_{:05d}.npy'.format(idx)), xs)
        np.save(str(target_dir / 'y_{:05d}.npy'.format(idx)), ys)
        del xs, ys


def image_generator(npy_dir, batch_size):
    x_paths = sorted(npy_dir.glob('x_*.npy'))
    y_paths = sorted(npy_dir.glob('y_*.npy'))

    idx = 0
    x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    while True:
        for x_path, y_path in zip(x_paths, y_paths):
            x_part = np.load(x_path)
            y_part = np.load(y_path)
            for x, y in zip(x_part, y_part):
                x_batch[idx] = x
                y_batch[idx] = y
                if idx + 1 == batch_size:
                    yield x_batch, y_batch
                idx = (idx + 1) % batch_size
            del x_part, y_part


def window_generator(npy_dir, batch_size):
    x_paths = sorted(npy_dir.glob('x_*.npy'))
    y_paths = sorted(npy_dir.glob('y_*.npy'))

    idx, timesteps = 0, 30
    x_batch = np.zeros((batch_size, timesteps, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    while True:
        for x_path, y_path in zip(x_paths, y_paths):
            x_part = np.load(x_path)
            y_part = np.load(y_path)
            for x, y in zip(x_part, y_part):
                x_batch[idx] = x
                y_batch[idx] = y
                if idx + 1 == batch_size:
                    yield x_batch, y_batch
                idx = (idx + 1) % batch_size
            del x_part, y_part



if __name__ == '__main__':
    DATASET = Path('~/dataset/').expanduser()
    DIRS = [x for x in DATASET.iterdir() if x.is_dir()]

    TRAIN_DIRS = DATASET[:-1]
    VAL_DIRS = DATASET[-1:]

    IMAGE_TRAIN = Path('npy/image_train')
    IMAGE_VAL = Path('npy/image_val')
    WINDOW_TRAIN = Path('npy/window_train')
    WINDOW_VAL = Path('npy/window_val')

    for folder in [IMAGE_TRAIN, IMAGE_VAL, WINDOW_TRAIN, WINDOW_VAL]:
        folder.mkdir(parents=True, exist_ok=True)

    gen_image_npy(TRAIN_DIRS, IMAGE_TRAIN, 25000)
    gen_image_npy(VAL_DIRS, IMAGE_VAL, 5000)
    gen_window_npy(TRAIN_DIRS, WINDOW_TRAIN, 25000, 30)
    gen_window_npy(VAL_DIRS, WINDOW_VAL, 5000, 30)

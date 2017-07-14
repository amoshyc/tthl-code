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
        windows = [(video_dir, i, i + timesteps) for i in range(n_frames - timesteps + 1)]
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
    

if __name__ == '__main__':
    dataset = Path('~/dataset/').expanduser()

    image_train = Path('npy/image_train')
    image_val = Path('npy/image_val')
    window_train = Path('npy/window_train')
    window_val = Path('npy/window_val')

    for folder in [image_train, image_val, window_train, window_val]:
        folder.mkdir(parents=True, exist_ok=True)

    # gen_image_npy([dataset / 'video01'], image_val, 5000)
    # gen_image_npy([dataset / 'video00'], image_train, 25000)
    gen_window_npy([dataset / 'video00'], window_train, 2500, 30)
    # gen_window_npy([dataset / 'video01'], window_val, 5000, 30)



# def image_generator(video_dirs, batch_size):
#     # Get all paths
#     x_all = []
#     y_all = []
#     for video_dir in video_dirs:
#         imgs = sorted((video_dir / 'frames/').iterdir())
#         label = json.load((video_dir / 'label.json').open())['label']
#         x_all.extend(imgs)
#         y_all.extend(label)

#     indices = np.random.permutation(len(x_all))
#     x_use = [x_all[i] for i in indices]
#     y_use = [y_all[i] for i in indices]

#     x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
#     y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

#     while True:
#         for i, img_path in enumerate(x_use):
#             idx = i % batch_size
#             x_batch[idx] = read_img(img_path)
#             y_batch[idx] = y_use[i]

#             if idx == batch_size - 1:
#                 yield (x_batch, y_batch)


# def window_generator(video_dirs, batch_size, timesteps):
#     x_batch = np.zeros((batch_size, timesteps, 224, 224, 3), dtype=np.float32)
#     y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

#     idx = 0
#     while True:
#         for video_dir in video_dirs:
#             xs = sorted((video_dir / 'frames/').iterdir())
#             ys = json.load((video_dir / 'label.json').open())['label']

#             # [i, i + timesteps)
#             windows = [(i, i + timesteps)
#                        for i in range(len(xs) - timesteps + 1)]
#             random.shuffle(windows)

#             for (s, e) in windows:
#                 for f in range(s, e):
#                     x_batch[idx][f - s] = read_img(xs[f])
#                 y_batch[idx] = ys[e - 1]

#                 if idx == batch_size - 1:
#                     yield x_batch, y_batch
#                 idx = (idx + 1) % batch_size


# def window_npy_generator(video_dirs, batch_size, timesteps):
#     x_batch = np.zeros((batch_size, timesteps, 224, 224, 3), dtype=np.float32)
#     y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

#     idx = 0
#     while True:
#         for video_dir in video_dirs:
#             xs = np.load(str(video_dir / 'imgs.npy'))
#             ys = json.load((video_dir / 'label.json').open())['label']

#             # [i, i + timesteps)
#             windows = [(i, i + timesteps)
#                        for i in range(len(xs) - timesteps + 1)]
#             random.shuffle(windows)

#             for (s, e) in windows:
#                 for f in range(s, e):
#                     x_batch[idx][f - s] = xs[f]
#                 y_batch[idx] = ys[e - 1]

#                 if idx == batch_size - 1:
#                     yield x_batch, y_batch
#                 idx = (idx + 1) % batch_size

#             xs = None

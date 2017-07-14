import json
import random
from pathlib import Path

import numpy as np
from util import read_img

def image_generator(video_dirs, n_samples, batch_size):
    # Get all paths
    x_all = []
    y_all = []
    for video_dir in video_dirs:
        imgs = sorted((video_dir / 'frames/').iterdir())
        label = json.load((video_dir / 'label.json').open())['label']
        x_all.extend(imgs)
        y_all.extend(label)
    
    indices = np.random.permutation(len(x_all))[:n_samples]
    x_use = [x_all[i] for i in indices]
    y_use = [y_all[i] for i in indices]

    x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    while True:
        for i, img_path in enumerate(x_use):
            idx = i % batch_size
            x_batch[idx] = read_img(img_path)
            y_batch[idx] = y_use[i]

            if idx == batch_size - 1:
                yield (x_batch, y_batch)

def window_generator(video_dirs, n_samples, batch_size, timesteps):
    x_batch = np.zeros((batch_size, timesteps, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    idx = 0
    while True:
        for video_dir in video_dirs:
            xs = np.load(str(video_dir / 'imgs.npy'))
            ys = json.load((video_dir / 'label.json').open())['label']

            # [i, i + timesteps)
            windows = [(i, i + timesteps) for i in range(len(xs) - timesteps + 1)]
            random.shuffle(windows)[:samples]

            for (s, e) in windows:
                for f in range(s, e):
                    x_batch[idx][f - s] = xs[f]
                y_batch[idx] = ys[e - 1]

                if idx == batch_size - 1:
                    yield x_batch, y_batch
                idx = (idx + 1) % batch_size

            xs = None




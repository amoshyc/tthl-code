import json
import random
from pathlib import Path
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from config import *


class ImageNpzCreator(object):
    '''Create `image_train.npz`, `image_val.npz` at target_dir from image extracted from video_dirs
    '''

    def __init__(self, n_train=100, n_val=20, fps=1,
                 target_dir=Path('./npz/')):
        self.n_train = n_train
        self.n_val = n_val
        self.fps = fps
        self.target_dir = target_dir

    def extract_data(self, video_dir):
        video = VideoFileClip(str(video_dir / 'video.mp4'))
        info = json.load((video_dir / 'info.json').open())
        n_frames = int(video.duration) * self.fps

        label = np.zeros(n_frames, dtype=np.uint8)
        for s, e in zip(info['starts'], info['ends']):
            fs = round(s * self.fps)
            fe = round(e * self.fps)
            label[fs:fe + 1] = 1

        data = [(video, f, label[f]) for f in range(n_frames)]
        return data

    def gen_npz(self, data, name):
        n = len(data)
        xs = np.zeros((n, 224, 224, 3), dtype=np.float32)
        ys = np.zeros((n, 1), dtype=np.uint8)

        for i, (video, f, y) in enumerate(tqdm(data, ascii=True)):
            img = video.get_frame(f / self.fps)
            xs[i] = scipy.misc.imresize(img, (224, 224))
            ys[i] = y

        npz_path = self.target_dir / '{}.npz'.format(name)
        np.savez(npz_path, xs=xs, ys=ys)
        del xs, ys

    def fit1(self, video_dirs):
        train, val = [], []
        for video_dir in video_dirs:
            data = self.extract_data(video_dir)
            pivot = round(
                (self.n_train) / (self.n_train + self.n_val) * len(data))
            train.extend(data[:pivot])
            val.extend(data[pivot:])

        train = random.sample(train, k=self.n_train)
        val = random.sample(val, k=self.n_val)

        self.target_dir.mkdir(exist_ok=True, parents=True)
        self.gen_npz(train, 'image_train')
        self.gen_npz(val, 'image_val')

    def fit2(self, video_dirs):
        pivot = -2
        train, val = [], []
        for video_dir in video_dirs[:pivot]:
            train.extend(self.extract_data(video_dir))
        for video_dir in video_dirs[pivot:]:
            val.extend(self.extract_data(video_dir))

        train = random.sample(train, k=self.n_train)
        val = random.sample(val, k=self.n_val)

        self.target_dir.mkdir(exist_ok=True, parents=True)
        self.gen_npz(train, 'image_train')
        self.gen_npz(val, 'image_val')


class WindowNpzCreator(object):
    '''Create `window_train.npz`, `window_val.npz`  at target_dir from windows extracted from video_dirs
    '''

    def __init__(self,
                 n_train=None,
                 n_val=None,
                 fps=1,
                 timesteps=5,
                 overlap=4,
                 target_dir=Path('./npz/')):
        self.n_train = n_train or 100
        self.n_val = n_val or 20
        self.fps = fps
        self.timesteps = timesteps
        self.overlap = overlap
        self.target_dir = target_dir

    def extract_windows(self, video_dir):
        video = VideoFileClip(str(video_dir / 'video.mp4'))
        info = json.load((video_dir / 'info.json').open())
        n_frames = int(video.duration) * self.fps
        timesteps = self.timesteps
        overlap = self.overlap

        label = np.zeros(n_frames, dtype=np.uint8)
        for s, e in zip(info['starts'], info['ends']):
            fs = round(s * self.fps)
            fe = round(e * self.fps)
            label[fs:fe + 1] = 1

        windows = [(video, f - timesteps, f, label[f - 1])
                   for f in range(timesteps, n_frames, timesteps - overlap)]
        print(video_dir, int(video.duration), len(windows))
        return windows

    def gen_npz(self, windows, name):
        n = len(windows)
        xs = np.zeros((n, self.timesteps, 224, 224, 3), dtype=np.float32)
        ys = np.zeros(n, dtype=np.uint8)
        for i, (video, s, e, y) in enumerate(tqdm(windows, ascii=True)):
            for j in range(e - s):
                img = video.get_frame((s + j) / self.fps)
                xs[i][j] = scipy.misc.imresize(img, (224, 224))
            ys[i] = y

        n_1 = np.count_nonzero(ys)
        print('{}: {} / {} = {}'.format(name, n_1, len(ys), n_1 / len(ys)))
        npz_path = self.target_dir / '{}.npz'.format(name)
        np.savez(npz_path, xs=xs, ys=ys)
        del xs, ys

    def fit1(self, video_dirs):
        train, val = [], []
        for video_dir in video_dirs:
            windows = self.extract_windows(video_dir)
            random.shuffle(windows)
            pivot = round(
                (self.n_train) / (self.n_train + self.n_val) * len(windows))
            train.extend(windows[:pivot])
            val.extend(windows[pivot:])

        train = random.sample(train, k=self.n_train)
        val = random.sample(val, k=self.n_val)

        self.target_dir.mkdir(exist_ok=True, parents=True)
        self.gen_npz(train, 'window_train')
        self.gen_npz(val, 'window_val')

    def fit2(self, video_dirs):
        pivot = -2
        train, val = [], []
        for video_dir in video_dirs[:pivot]:
            train.extend(self.extract_windows(video_dir))
        for video_dir in video_dirs[pivot:]:
            val.extend(self.extract_windows(video_dir))

        train = random.sample(train, k=self.n_train)
        val = random.sample(val, k=self.n_val)

        self.target_dir.mkdir(exist_ok=True, parents=True)
        self.gen_npz(train, 'window_train')
        self.gen_npz(val, 'window_val')

def main():
    # gen = ImageNpzCreator(n_train=10000, n_val=2000, fps=2, target_dir=d2_dir)
    # gen.fit2(video_dirs)

    gen = WindowNpzCreator(
        n_train=10000, n_val=2000, fps=2, timesteps=4, overlap=3, target_dir=d2_dir)
    gen.fit2(video_dirs)


if __name__ == '__main__':
    main()

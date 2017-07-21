import json
import random
from pathlib import Path
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip
from tqdm import tqdm

class WindowNpyGenerator(object):
    def __init__(self, n_train=None, n_val=None, fps=1, timesteps=5, overlap=4, target_dir=None):
        self.video_dirs = []
        self.n_train = n_train or 100
        self.n_val = n_val or 20
        self.fps = fps
        self.timesteps = timesteps
        self.overlap = overlap
        self.target_dir = target_dir or Path('./npy/') 

    def extract_windows(video_dir):
        video = VideoFileClip(str(video_dir / 'video.mp4'))
        info = json.load((video_dir / 'info.json').open())
        n_frames = int(video.duration) * self.fps
        timesteps = self.timesteps
        overlap = self.overlap

        label = np.zeros(n_frames, dtype=np.uint8)
        for s, e in zip(info['starts'], info['ends']):
            fs = round(s * fps)
            fe = round(e * fps)
            label[fs:fe + 1] = 1

        windows = [(video, f - timesteps, f, label[f - 1])
                for f in range(timesteps, n_frames, n_frames - overlap)]
        return windows

    def gen_npz(self, windows, chunk_size=10000):  
        for i in tqdm(range(0, len(windows), chunk_size), desc='chunk'):
            chunk_s, chunk_e = i, min(i + chunk_size, len(windows))
            chunk = windows[chunk_s:chunk_e]

            xs = np.zeros((len(chunk), self.timesteps, 224, 224, 3), dtype=np.float32)
            ys = np.zeros(len(chunk), dtype=np.uint8)
            for i, (video, s, e, y) in enumerate(tqdm(chunk), desc='data'):
                for j in range(e - s):
                    img = video.get_frame((s + j) / fps)
                    xs[i][j] = scipy.misc.imresize(img, (224, 224))
                ys[i] = y

            npz_path = self.target_dir / '{:04d}.npy'.format(i // chunk_size)
            np.savez(npz_path, xs=xs, ys=ys)

    def fit(self, video_dirs):
        for video_dir in video_dirs:
            windows = self.extract_windows(video_dir)
            random.shuffle(windows)
            pivot = (self.n_train) / (self.n_train + self.n_val) * len(windows)
            train.extend(windows[:pivot])
            val.extend(windows[pivot:])
        
        train = random.sample(train, k=self.n_train)
        val = random.sample(val, k=self.n_val)
        
        self.target_dir.mkdir(exist_ok=True)
        self.gen_npz(train)
        self.gen_npz(val)

    def flow(self, batch_size=80):
        npzs = sorted(self.target_dir.glob('*.npz'))

        idx = 0
        x_batch = np.zeros((batch_size, self.timesteps, 224, 224, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

        while True:
            for npz in npzs:
                for x, y in zip(npz['xs'], npz['ys']):
                    x_batch[idx] = x
                    y_batch[idx] = y
                    if idx + 1 == batch_size:
                        yield x_batch, y_batch
                    idx = (idx + 1) % batch_size


if __name__ == '__main__':
    dataset = Path('~/tthl-dataset/').expanduser()
    video_dirs = sorted(dataset.glob('video*/'))

    gen = WindowNpyGenerator()
    gen.fit(video_dirs[:3]) 
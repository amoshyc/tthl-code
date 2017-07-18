import json
import random
from pathlib import Path
import numpy as np
from scipy.misc import imresize
from moviepy.editor import VideoFileClip


def window_generator(video_dirs, n_samples, batch_size, timesteps):
    videos = [VideoFileClip(str(x / 'video.mp4')) for x in video_dirs]
    labels = [
        json.load((x / 'label.json').open())['label'] for x in video_dirs
    ]

    windows = []
    for video_id, (video, label) in enumerate(zip(videos, labels)):
        fps, dur = video.fps, video.duration
        n_frames = round(dur * fps)
        cur_windows = [(video_id, t - 1, label[t - 1])
                       for t in range(timesteps, n_frames)]
        windows.extend(cur_windows)

    windows = random.sample(windows, k=n_samples)

    idx = 0
    x_batch = np.zeros((batch_size, timesteps, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    for video_id, e, label in windows:
        for t in range(timesteps):
            time = (e - t) / videos[video_id].fps
            img = videos[video_id].get_frame(time)
            x_batch[idx][timesteps - 1 - t] = imresize(img, (224, 224))
        y_batch[idx] = label

        if idx + 1 == batch_size:
            yield x_batch, y_batch
        idx = (idx + 1) % batch_size

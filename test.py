import json
import random
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip


def window_generator(video_dirs, n_samples, batch_size, timesteps):
    videos = [VideoFileClip(str(x / 'video.mp4')) for x in video_dirs]

    windows = []
    for video_id, (video, label) in enumerate(zip(videos, labels)):
        fps, dur = video.fps, video.duration
        n_frames = round(dur * fps)
        start = (e - timesteps) / fps
        end = (e) / fps
        label = json.load((x / 'label.json').open())['label']
        cur_windows = [(video_id, start, end, label[e - 1])
                       for e in range(timsteps, n_frames)]
        window.extend(cur_windows)

    windows = random.sample(windows, k=n_samples)

    idx = 0
    x_batch = np.zeros((batch_size, timestep, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    for video_id, s, e, label in windows:
        clip = list(videos[video_id].subclip(s, e).iter_frame())
        assert len(clip) == timesteps, 'len(clip) != timesteps'

        x_batch[idx] = np.array(clip, dtype=np.float32)
        y_batch[idx] = label

        if idx + 1 == batch_size:
            yield x_batch, y_batch
        idx = (idx + 1) % batch_size

import argparse
from pathlib import Path

import numpy as np
import scipy

import keras
from keras.models import load_model

from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

from find_players import find_players


def main():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Video Highlight')
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('video', type=str, help='Path to video to highlight')
    parser.add_argument('--out', '-o', type=str, default='./hl.mp4', help='output name')
    parser.add_argument('--fps', type=int, default=2, help='fps')
    parser.add_argument('--itv', type=int, default=6, help='interval of adjusting')
    parser.add_argument('--bs', type=int, default=30, help='batch size')
    args = parser.parse_args()
    # yapf: enable

    print('Loading model & video', end='...')
    model = load_model(args.model)
    video = VideoFileClip(args.video)
    print('ok')

    n_frames = int(video.duration) * args.fps
    xs = np.zeros((n_frames, 224, 224, 3), dtype=np.float32)
    for f in tqdm(range(n_frames), desc='Loading Video Frames', ascii=True):
        img = video.get_frame(f / args.fps)
        xs[f] = scipy.misc.imresize(img, (224, 224))

    p1, p2 = find_players(xs, save=False)

    # Predicting
    pred = model.predict([xs, p1, p2], args.bs, verbose=1)
    pred = pred.round().astype(np.uint8).flatten()

    print(pred[:500])

    for i in range(n_frames - args.itv):
        s, t = i, i + args.itv
        if pred[s] == 1 and pred[t - 1] == 1:
            pred[s:t] = 1

    diff = np.diff(np.concatenate([[0], pred, [1]]))
    starts = (diff == +1).nonzero()[0] / args.fps
    ends = (diff == -1).nonzero()[0] / args.fps
    segs = [video.subclip(s, e) for s, e in zip(starts, ends)]
    out = concatenate_videoclips(segs)
    out.write_videofile(args.out, fps=video.fps, threads=4, audio=True)


if __name__ == '__main__':
    main()

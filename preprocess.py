import json
import argparse
import subprocess
import shutil
import random
from pathlib import Path
from myutils import *
import numpy as np
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('ds', help='dataset', type=str, default='~/ds/')
args = parser.parse_args()

dataset = Path(args.ds).expanduser()
video_dirs = sorted(dataset.glob('video*/'))


def download_and_write_frames():
    target_hl_dir = Path('./tmp/hl/')
    target_hl_dir.mkdir(parents=True, exist_ok=True)
    target_non_dir = Path('./tmp/non/')
    target_non_dir.mkdir(parents=True, exist_ok=True)

    for video_id, video_dir in enumerate(video_dirs):
        # download
        video = video_dir / 'video.mp4'
        info = json.load((video_dir / 'info.json').open())
        url, starts, ends = info['video_src'], info['starts'], info['ends']
        download_cmd = f'youtube-dl --no-playlist -f 18 -o {video} {url}'
        subprocess.run(download_cmd, shell=True)

        # highlight video
        hl_video_path = target_hl_dir / f'{video_id:02d}.mp4'
        hl_frame_fmt = target_hl_dir / f'{video_id:02d}_%05d.jpg'
        video_concat_segments(video, hl_video_path, starts, ends)
        video_write_frames(hl_video_path, hl_frame_fmt, fps=2)

        # non-highlight video
        ss = [0] + info['ends']
        es = info['starts'] + [video_get_duration(video)]
        ss = [t + 0.25 for t in ss]
        es = [t - 0.25 for t in es]
        non_video_path = target_non_dir / f'{video_id:02d}.mp4'
        non_frame_fmt = target_non_dir / f'{video_id:02d}_%05d.jpg'
        video_concat_segments(video, non_video_path, ss, es)
        video_write_frames(non_video_path, non_frame_fmt, fps=2)


def train_val_split():
    hl_dir = Path('./data/hl/')
    non_dir = Path('./data/non/')

    # yapf: disable
    target_dirs = [
        Path('./train/hl/'),
        Path('./train/non/'),
        Path('./val/hl/'),
        Path('./val/non/')
    ]
    source_imgs = [
        [x for x in hl_dir.iterdir() if int(x.stem[:2]) < 9],
        [x for x in non_dir.iterdir() if int(x.stem[:2]) < 9],
        [x for x in hl_dir.iterdir() if int(x.stem[:2]) >= 9],
        [x for x in non_dir.iterdir() if int(x.stem[:2]) >= 9]
    ] # yapf: enable

    for folder, imgs in zip(target_dirs, source_imgs):
        folder.mkdir(parents=True, exist_ok=True)
        for img in tqdm(list(imgs), ascii=True):
            shutil.copy(str(img), str(folder))


def gen_npz():
    npz_dir = Path('./npz')
    npz_dir.mkdir(exist_ok=True)

    for folder in [Path('./train/'), Path('./val/')]:
        imgs = sorted(list(folder.glob('*/*.jpg')))
        random.shuffle(imgs)
        classes = [folder / 'non', folder / 'hl'] # non=0, hl=1

        n_samples = len(imgs)
        target_size = (224, 224)
        xs = np.zeros((n_samples, *target_size, 3), dtype=np.float32)
        ys = np.zeros((n_samples, ), dtype=np.uint8)
        for i, img_path in enumerate(tqdm(imgs, ascii=True)):
            img = Image.open(img_path)
            img = img.resize(target_size)
            xs[i] = np.array(img)
            ys[i] = classes.index(img_path.parent)

        npz_path = npz_dir / f'{folder.stem}.npz'
        np.savez(npz_dir / folder.stem, xs=xs, ys=ys)


if __name__ == '__main__':
    download_and_write_frames()
    train_val_split()
    gen_npz()

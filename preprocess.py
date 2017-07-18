import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips  # requires ffmpeg

from utils import read_img


def process(video_dir, gen_frames=False, gen_highlight=False, gen_label=False):
    """Generating frames, highlights and labels of the video.
    Frames is written as `video_dir/%08d.jpg`
    Highlight is generated as `highlight.mp4`
    Label is written back to `info.json`
    
    Arguments:
    video_dir:  A pathlib.Path object pointing to the folder of the target video. 
                The directory should contain `video.mp4` and `info.json`.
    gen_frames, gen_highlight,  gen_label: 
                Controls whether to generate frames, highlight, label repectively.
                Default to True
    """

    video_path = video_dir / 'video.mp4'
    hl_path = video_dir / 'highlight.mp4'
    info_path = video_dir / 'info.json'
    label_path = video_dir / 'label.json'
    frame_dir = video_dir / 'frames/'
    frame_fmt = frame_dir / '%08d.jpg'
    frame_dir.mkdir(exist_ok=True)

    video = VideoFileClip(str(video_path))
    info = json.load(info_path.open())

    if gen_frames:
        print('Generating frames')
        video.write_images_sequence(str(frame_fmt))

    if gen_highlight:
        print('Generating highlight')
        clips = [
            video.subclip(s, e) for s, e in zip(info['starts'], info['ends'])
        ]
        hl = concatenate_videoclips(clips)
        hl.write_videofile(str(hl_path), threads=3)

    if gen_label:
        print('Generating label...', end='')
        n_frames = len(list(frame_dir.iterdir()))
        label = {'label': [0 for _ in range(n_frames)]}
        for s, e in zip(info['starts'], info['ends']):
            fs = round(s * video.fps)
            fe = round(e * video.fps)
            for i in range(fs, fe + 1):
                label['label'][i] = 1
        with label_path.open('w') as f:
            json.dump(label, f, ensure_ascii=False)
        print('ok')


def process_all():
    dataset_dir = Path('~/dataset').expanduser().resolve()
    video_dirs = [x for x in dataset_dir.iterdir() if x.is_dir()]
    for i, video_dir in enumerate(video_dirs):
        print(video_dir, '({}/{})'.format(i + 1, len(video_dirs)))
        process(video_dir, gen_frames=True, gen_label=True)
        print('*' * 50)


if __name__ == '__main__':
    process_all()
    # video_dir = Path('~/dataset/video05').expanduser().resolve()
    # process(video_dir, gen_frames=True, gen_highlight=False, gen_label=True)

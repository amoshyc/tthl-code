import json

import numpy as np
import scipy

from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

def extract_images(video_dirs, fps):
    for i, video_dir in enumerate(video_dirs):
        video = VideoFileClip(str(video_dir / 'video.mp4'))
        info = json.load((video_dir / 'info.json').open())

        segs = [video.subclip(s, e) for s, e in zip(info['starts'], info['ends'])]
        hl = concatenate_videoclips(segs)
        hl.write_videofile(str(video_dir / 'gt.mp4'), threads=5, audio=False)

        frame_dir = (video_dir / 'frames/')
        frame_dir.mkdir(exist_ok=True)
        hl.write_images_sequence(str(frame_dir / '%04d.png'), fps=fps)

def main():
    from config import video_dirs
    extract_images(video_dirs[1:], 2)

if __name__ == '__main__':
    main()
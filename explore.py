import json
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip


def get_video_info(video_dir):
    info = json.load((video_dir / 'info.json').open())
    path = str(video_dir / 'video.mp4')
    video = VideoFileClip(path)
    res = {
        'len': video.duration,
        'hl': sum(e - s for s, e in zip(info['starts'], info['ends'])),
    }
    return res


if __name__ == '__main__':
    dataset = Path('~/dataset')
    all_info = [get_video_info(x) for x in dataset.iter_dir() if x.is_dir()]
    total_len = sum(info['len'] for info in all_info)
    total_hl = sum(info['hl'] for info in all_info)
    print('total len', total_len)
    print('total hl', total_hl)
    print(total_hl / total_len)
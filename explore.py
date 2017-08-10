import json
from pathlib import Path
from pprint import pprint
import numpy as np
from moviepy.editor import VideoFileClip


def get_video_info(video_dir):
    info = json.load((video_dir / 'info.json').open())
    path = str(video_dir / 'video.mp4')
    video = VideoFileClip(path)
    res = {
        'len': video.duration,
        'hl': sum(e - s for s, e in zip(info['starts'], info['ends'])),
        'fps': video.fps,
        'seg': len(info['starts'])
    }
    return res


if __name__ == '__main__':
    from config import video_dirs

    all_info = [get_video_info(video_dir) for video_dir in video_dirs]

    for key in ['len', 'hl', 'fps', 'seg']:
        print(key.upper())
        print('\n'.join([f'{info[key]:10.2f}' for info in all_info]))
        # print('\n'.join([str(info[key]) for info in all_info]))
        print('-' * 60)

    total_len = sum(info['len'] for info in all_info)
    total_hl = sum(info['hl'] for info in all_info)
    print('Total')
    print('len :', total_len)
    print('hl  :', total_hl)
    print('prop:', total_hl / total_len)
    print('seg :', sum(info['seg'] for info in all_info))

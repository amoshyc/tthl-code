import json
from pathlib import Path
from subprocess import run

dataset = Path('~/dataset/').expanduser()
video_dirs = [x for x in dataset.iterdir() if x.is_dir()]
for video_dir in video_dirs:
    url = json.load((video_dir / 'info.json').open())['video_src']
    run(['youtube-dl', '-f', '18', '-o', str(video_dir / 'video.mp4'), url])
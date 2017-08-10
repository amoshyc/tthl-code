import json
from pathlib import Path
from subprocess import run
from config import video_dirs

# run(['rm', '-rf', '{}/**/*.mp4'.format(dataset)])

for i, video_dir in enumerate(video_dirs):
    print('{} ({} / {})'.format(video_dir, i + 1, len(video_dirs)))
    url = json.load((video_dir / 'info.json').open())['video_src']
    run(['youtube-dl', '--no-playlist', '-f', '18', '-o', str(video_dir / 'video.mp4'), url])
    print('-' * 60)
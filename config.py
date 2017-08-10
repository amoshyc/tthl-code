from pathlib import Path

dataset = Path('~/tthl-dataset/').expanduser()
video_dirs = sorted(dataset.glob('video*/'))

d1_dir = Path('./d1/')
d2_dir = Path('./d2/')
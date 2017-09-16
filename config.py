from pathlib import Path

dataset = Path('../ds/').expanduser()
video_dirs = sorted(dataset.glob('video*/'))

d1_dir = Path('./d1/')
d2_dir = Path('./d2/')

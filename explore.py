import json
import argparse
from pathlib import Path

import pandas as pd
from scipy.misc import imread
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from myutils import video_get_fps, video_get_duration

parser = argparse.ArgumentParser()
parser.add_argument('ds', help='dataset', type=str)
args = parser.parse_args()

video_dirs = [x for x in Path(args.ds).iterdir() if x.is_dir()]
video_dirs = sorted(list(video_dirs))

df = pd.DataFrame(
    index=range(len(video_dirs)),
    columns=['fps', 'len', 'hl', 'non'])
for i, folder in enumerate(video_dirs):
    video = folder / 'video.mp4'
    info = json.load((folder / 'info.json').open())
    ss, es = info['starts'], info['ends']

    df['fps'][i] = video_get_fps(video)
    df['len'][i] = video_get_duration(video)
    df['hl'][i]= sum(float(e) - float(s) for s, e in zip(ss, es))
    df['non'][i] = df['len'][i] - df['hl'][i]

print(df)
print('---')
print(df[['len', 'hl', 'non']].sum(axis=0))

fig = plt.figure(1, (16, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.1)

for i in range(11):
    img = imread(f'./tmp/hl/{i:02d}_00001.jpg')
    grid[i].imshow(img)
    grid[i].axis('off')
grid[-1].axis('off')

plt.tight_layout()
plt.savefig('explore.svg')

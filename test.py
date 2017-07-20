import pathlib
import numpy as np
import skvideo.io
from moviepy.editor import VideoFileClip

path = pathlib.Path('~/dataset/video00/video.mp4').expanduser()
# videodata = skvideo.io.vread(str(path))
# _ = input(':')
# print(videodata.shape)

clip = VideoFileClip(str(path))
frames = np.array(list(clip.iter_frames()))
_ = input(':')
print(frames.shape)
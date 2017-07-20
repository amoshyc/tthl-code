import pathlib
import numpy as np

path = pathlib.Path('~/dataset/video00/video.mp4').expanduser()

# import skvideo.io
# videodata = skvideo.io.vread(str(path))
# _ = input(':')
# print(videodata.shape)

from moviepy.editor import VideoFileClip
clip = VideoFileClip(str(path))
_ = input(':')
print(clip.duration, clip.fps)
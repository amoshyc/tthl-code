import pathlib
import numpy as np

path = pathlib.Path('~/dataset/video00/video.mp4').expanduser()

# import skvideo.io
# # videodata = skvideo.io.vread(str(path))
# # _ = input(':')
# # print(videodata.shape)

# from moviepy.editor import VideoFileClip
# clip = VideoFileClip(str(path))
# frames = np.array(list(clip.iter_frames()))
# _ = input(':')
# print(frames.shape)
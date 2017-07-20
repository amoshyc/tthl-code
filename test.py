import pathlib
import skvideo.io
path = pathlib.Path('~/dataset/video00/video.mp4').expanduser()
videodata = skvideo.io.vread(str(path))
print(videodata.shape)
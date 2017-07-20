import pathlib
import skvideo.io
path = pathlib.Path('~/dataset/video00/video.mp4').expanduser()
videodata = skvideo.io.vread(str(path), width=224, height=224)
_ = input()
print(videodata.shape)
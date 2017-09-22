import subprocess
import pathlib
from datetime import datetime


def video_concat_segments(infile, outfile, ss, ee):
    assert len(ss) == len(ee), 'number of ss != number of ee'
    for s, e in zip(ss, ee):
        assert s <= e, f'start({s}) should be <= end({e})'

    # ffmpeg -i test.mp4
    # -filter_complex "\
    #   [0]trim=1:10,setpts=PTS-STARTPTS[v0]; \
    #   [0]trim=10:20,setpts=PTS-STARTPTS[v1]; \
    #   [v0][v1]concat=n=2[out]"
    # -map [out] out.mp4

    n = len(ss)
    inputs = [
        f'[0]trim={ss[i]}:{ee[i]},setpts=PTS-STARTPTS[v{i}]' for i in range(n)
    ]
    streams = ''.join([f'[v{i}]' for i in range(n)])
    concat = f'concat=n={n}[out]'
    filter_cmd = ';'.join([*inputs, streams + concat])

    cmd = f'ffmpeg -i {infile} -filter_complex "{filter_cmd}" -map [out] {outfile} -y'
    subprocess.run(cmd, shell=True)


def video_write_frames(infile, outfile='./frames/%05d.jpg', fps=1):
    # ffmpeg -i out.mp4 -vf fps="fps=1" outfile
    pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -i {infile} -vf fps="fps={fps}" {outfile}'
    subprocess.run(cmd, shell=True)


def video_get_duration(video):
    cmd = f'ffprobe -i {video} -show_entries format=duration -v quiet -of csv="p=0"'
    out = subprocess.check_output(cmd, shell=True)
    return float(str(out, 'utf-8'))

def video_get_fps(video):
    cmd = f'ffprobe -i {video} -show_entries stream=r_frame_rate -v quiet -of csv="p=0"'
    out = subprocess.check_output(cmd, shell=True)
    res = str(out.split()[0], 'utf-8').split('/')
    nem, dem = float(res[0]), float(res[1])
    return nem / dem

def get_callbacks(name):
    from keras.callbacks import Callback, ModelCheckpoint, CSVLogger

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = pathlib.Path('./log/') / f'{name}_{now}'
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(log_dir / 'log.csv')
    ckpt_path = str(log_dir / '{val_binary_accuracy:.3f}_{epoch:02d}.h5')

    return [CSVLogger(csv_path), ModelCheckpoint(ckpt_path)]

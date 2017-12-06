import subprocess
import pathlib
from datetime import datetime

from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use('seaborn')


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

def get_callbacks(name, acc='val_binary_accuracy'):
    from keras.callbacks import Callback, ModelCheckpoint, CSVLogger

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = pathlib.Path('./log/') / f'{name}_{now}'
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(log_dir / 'log.csv')
    ckpt_path = str(log_dir / ('{' + acc + ':.3f}_{epoch:02d}.h5'))

    return [CSVLogger(csv_path), ModelCheckpoint(ckpt_path)]


def pr_curve(y_true, y_score, color='teal'):
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(dpi=100)
    ax.step(recall, precision, where='post', alpha=0.5, color='k')
    ax.fill_between(recall, precision, step='post', alpha=0.5, color=color)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('AP = {:.3f}'.format(ap))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # plt.show()
    return fig, ax
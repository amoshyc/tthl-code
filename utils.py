from datetime import datetime
import numpy as np
from keras.preprocessing import image
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def read_img(path, target_size=(224, 224)):
    pil = image.load_img(str(path), target_size=target_size)
    return image.img_to_array(pil)


def get_callbacks(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = 'log/{} ({}).csv'.format(name, now)
    weight_path = '/tmp/' + name + '_{epoch:02d}_{val_binary_accuracy:.3f}.h5'

    return [CSVLogger(log_path), ModelCheckpoint(filepath=weight_path)]

def sample(x, y, k=None):
    indices = np.random.permutation(len(x))[:k]
    x_res = [x[i] for i in indices]
    y_res = [y[i] for i in indices]
    return x_res, y_res

def split(x, y, k=None):
    n, idx = len(x), 0
    res = []
    for i in range(0, n, k):
        s, e = i, min(i + k, n)
        res.append(tuple(idx, x[s:e], y[s:e]))
        idx += 1
    return res

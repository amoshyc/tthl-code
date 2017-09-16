from datetime import datetime
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def get_callbacks(name):
    now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_path = 'log/{} ({}).csv'.format(name, now)
    weight_path = './tmp/' + name + '_{epoch:02d}_{val_binary_accuracy:.3f}.h5'

    return [CSVLogger(log_path), ModelCheckpoint(filepath=weight_path)]

def imgs_to_npz(paths, labels, target_path, size=(224, 224), channel_first=False):
    n_imgs = len(paths)

    xs = np.zeros((n_imgs, size[0], size[1], 3), dtype=np.float32)
    ys = np.array(labels, dtype=np.uint8)

    for i, path in enumerate(paths):
        img = Image.open(path).resize(size)
        xs[i] = np.array(img)

    if channel_first:
        xs = xs.tranpose([0, 3, 1, 2])

    np.savez(target_path, xs=xs, ys=ys)

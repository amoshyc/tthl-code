from datetime import datetime
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def get_callbacks(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = 'log/{} ({}).csv'.format(name, now)
    weight_path = '/tmp/{}_{epoch:02d}_{val_binary_accuracy:.3f}.h5'.format(name)

    return [
        CSVLogger(log_path),
        ModelCheckpoint(filepath=weight_path)
    ]

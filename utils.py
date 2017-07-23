from datetime import datetime
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def get_callbacks(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = 'log/{} ({}).csv'.format(name, now)
    weight_path = '/tmp/' + name + '_{epoch:02d}_{val_binary_accuracy:.3f}.h5'

    return [CSVLogger(log_path), ModelCheckpoint(filepath=weight_path)]

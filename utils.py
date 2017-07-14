from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def get_callbacks(name):
    log_path = '{}.log'.format(name)
    weight_path = '/tmp/{}_{epoch:02d}_{val_binary_accuracy:.3f}.h5'.format(name)

    return [
        CSVLogger(log_path),
        ModelCheckpoint(filepath=weight_path)
    ]

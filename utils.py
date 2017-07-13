from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def get_callbacks(name):
    return [
        CSVLogger(name + '.log'),
        ModelCheckpoint(filepath='/tmp/' + name +
                        '_epoch{epoch:02d}_{val_binary_accuracy:.3f}.h5')
    ]

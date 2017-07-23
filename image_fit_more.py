from sys import argv

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import load_model


def main():
    model = load_model(argv[1])

    train = np.load('npz/image_train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('npz/image_val.npz')
    x_val, y_val = val['xs'], val['ys']

    fit_arg = {
        'x': x_train, 
        'y': y_train,
        'batch_size': 100,
        'epochs': int(model[2]),
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': get_callbacks('cnn'),
    }
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()
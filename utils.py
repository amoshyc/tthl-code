from datetime import datetime
from keras.preprocessing import image
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def read_img(path, target_size=(224, 224)):
    pil = image.load_img(path, target_size=target_size)
    return image.img_to_array(pil)


def get_callbacks(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = 'log/{} ({}).csv'.format(name, now)
    weight_path = '/tmp/' + name + '_{epoch:02d}_{val_binary_accuracy:.3f}.h5'

    return [CSVLogger(log_path), ModelCheckpoint(filepath=weight_path)]

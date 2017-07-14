import json
from pathlib import Path

import numpy as np
from keras.preprocessing import image

def generator(video_dirs, n_samples, batch_size):
    # Get all paths
    x_all = []
    y_all = []
    for video_dir in video_dirs:
        imgs = sorted((video_dir / 'frames/').iterdir())
        info = json.load((video_dir / 'info.json').open())
        x_all.extend(imgs)
        y_all.extend(info['label'])
    
    indices = np.random.permutation(len(x_all))[:n_samples]
    x_use = [x_all[i] for i in indices]
    y_use = [y_all[i] for i in indices]

    x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    while True:
        for i, img_path in enumerate(x_use):
            idx = i % batch_size
            pil = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(pil)
            x_batch[idx] = img
            y_batch[idx] = y_use[i]

            if idx == batch_size - 1:
                yield (x_batch, y_batch)

dataset = Path('~/dataset/').expanduser().resolve()
video_dirs = sorted([x for x in dataset.iterdir() if x.is_dir()])
train_dirs = [(dataset / 'video01/')]
val_dirs = [(dataset / 'video00')]
train_gen = generator(train_dirs, n_train, batch_size)
val_gen = generator(val_dirs, n_val, batch_size)
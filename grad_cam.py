import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.misc import imsave, imread, imresize
from keras.models import load_model, Sequential
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_cam, overlay
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('model', help='model', type=str)
parser.add_argument('image_dir', help='image_dir', type=str)
parser.add_argument('output_dir', help='output_dir', type=str)
parser.add_argument('label', help='label', type=int)
args = parser.parse_args()

model = load_model(args.model)
paths = list(Path(args.image_dir).glob('*.jpg'))
paths = random.sample(paths, k=500)
output_dir = Path(args.output_dir)
(output_dir / '0').mkdir(parents=True, exist_ok=True)
(output_dir / '1').mkdir(parents=True, exist_ok=True)
layer_idx = utils.find_layer_idx(model, 'dense_2')

for path in tqdm(paths, ascii=True):
    img = imread(str(path))
    img = imresize(img, (224, 224))

    pred = np.argmax(model.predict(np.expand_dims(img, axis=0)))    
    grads = visualize_cam(model, layer_idx, filter_indices=args.label, seed_input=img, backprop_modifier=None)

    target_path = output_dir / str(pred) / f'{path.stem}_gcam.jpg'
    imsave(str(target_path), overlay(grads, img, 0.4))

# python grad_cam.py .\log\gc_2017-09-24_20-17-12\0.836_08.h5 .\tmp\hl\ .\cam\hl\ 1
# python grad_cam.py .\log\gc_2017-09-24_20-17-12\0.836_08.h5 .\tmp\non\ .\cam\non\ 0
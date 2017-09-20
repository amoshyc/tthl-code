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
output_dir.mkdir(parents=True, exist_ok=True)

for path in tqdm(paths, ascii=True):
    img = imread(str(path))
    img = imresize(img, (224, 224))
    layer_idx = utils.find_layer_idx(model, 'dense_2')

    grads = visualize_cam(model, layer_idx, filter_indices=args.label, seed_input=img, backprop_modifier=None)

    target_path = output_dir / f'{path.stem}_gcam.jpg'
    imsave(str(target_path), overlay(grads, img))

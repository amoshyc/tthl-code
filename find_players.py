import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from math import floor, ceil

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append('./tensorflow-models/')
sys.path.append('./tensorflow-models/object_detection')

import tensorflow as tf
import numpy as np
from scipy.misc import imresize, imsave

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tqdm import tqdm

def crop(x, bbox):
    y1 = floor(224 * bbox[0])
    x1 = floor(224 * bbox[1])
    y2 = ceil(224 * bbox[2])
    x2 = ceil(224 * bbox[3])
    return imresize(x[y1:y2, x1:x2], (100, 50))

def get_graph():
    MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    NUM_CLASSES = 90

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def find_players(xs, name='', save=True):
    detection_graph = get_graph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        player1 = []
        player2 = []
        none = np.zeros((100, 50, 3), dtype=np.float32)

        for x in tqdm(xs, ascii=True, desc=name):
            x = x.astype(np.uint8)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(x, axis=0)})

            person_indices = classes == 1 # person
            score_indices = scores > 0.50
            indices = person_indices & score_indices
            boxes = boxes[indices]
            scores = scores[indices]
            classes = classes[indices]
            num = indices.sum()

            if num == 0:
                player1.append(none)
                player2.append(none)
            elif num == 1:
                y1, x1, y2, x2 = boxes[0]
                player1.append(crop(x, boxes[0]))
                player2.append(none)
            else:
                player1.append(crop(x, boxes[0]))
                player2.append(crop(x, boxes[1]))

        player1 = np.array(player1, dtype=np.float32)
        player2 = np.array(player2, dtype=np.float32)

        if save:
            np.savez(f'npz/{name}.npz', p1=player1, p2=player2)

        return player1, player2


if __name__ == '__main__':
    train_data = np.load('npz/train.npz')
    val_data = np.load('npz/val.npz')
    x_train = train_data['xs']
    x_val = val_data['xs']

    find_players(x_train, 'players_train')
    find_players(x_val, 'players_val')

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
import scipy

import keras
from keras.models import load_model
from sklearn import metrics

def plot(ax, y_true, y_score, name):
    pr, re, th = metrics.precision_recall_curve(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)
    label = name + ' (AP = {:.3f})'.format(ap)
    ax.step(re, pr, where='post', label=label)


val = np.load('npz/val.npz')
x_val, y_val = val['xs'], val['ys']
players_val = np.load('npz/players_val.npz')
p1_val, p2_val = players_val['p1'], players_val['p2']


vgg16 = load_model('log/vgg16_2017-09-24_21-57-24/0.846_29.h5', compile=False)
vgg16_score = vgg16.predict(x_val, 80, verbose=1)
del vgg16

ens = load_model('log/ens2_2017-09-25_15-12-45/0.855_09.h5', compile=False)
ens_score = ens.predict(x_val, 80, verbose=1)
del ens

player16 = load_model('log/players_vgg16_2017-09-26_21-13-25/0.865_15.h5', compile=False)
player16_score = player16.predict([x_val, p1_val, p2_val], 80, verbose=1)
del player16

player19 = load_model('log/players_vgg19_2017-09-25_16-16-00/0.876_23.h5', compile=False)
player19_score = player19.predict([x_val, p1_val, p2_val], 80, verbose=1)
del player19


fig, ax = plt.subplots()
plot(ax, y_val, vgg16_score, 'vgg16')
plot(ax, y_val, ens_score, 'vgg16 + vgg19 + res50')
plot(ax, y_val, player16_score, 'vgg16 w/ players')
plot(ax, y_val, player19_score, 'vgg19 w/ players')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.legend(loc=3, fontsize=18)

fig.tight_layout()
fig.savefig('pr_curve.png', dpi=150)

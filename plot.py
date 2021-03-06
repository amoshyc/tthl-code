from sys import argv
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import pandas as pd

# python plot.py "$(ls -t log/* | head -n 1)" name

# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)


def plot_svg(log, name, n_epochs=None):
    df = pd.read_csv(log)
    graph = Path('./graph/')
    loss_path = graph / (name + '_loss.svg')
    acc_path = graph / (name + '_acc.svg')
    n_epochs = n_epochs or df.shape[0]

    print('min loss:', df['val_loss'].min())
    print('max acc :', df['val_binary_accuracy'].max())

    keys = ['loss', 'val_loss']
    ax = df[keys][:n_epochs].plot(kind='line')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss(binary crossentropy)')
    plt.savefig(str(loss_path))

    keys = ['binary_accuracy', 'val_binary_accuracy']
    ax = df[keys][:n_epochs].plot(kind='line')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    plt.savefig(str(acc_path))


if __name__ == '__main__':
    if len(argv) == 3:
        plot_svg(argv[1], argv[2])
    else:
        plot_svg(argv[1], argv[2], int(argv[3]))
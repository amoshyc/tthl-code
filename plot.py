from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import pandas as pd

# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)  

def plot_svg(log, name):
    df = pd.read_csv(log)
    graph = Path('./graph/')
    loss_path = graph / (name + '_loss.svg')
    acc_path = graph / (name + '_acc.svg')

    keys = ['loss', 'val_loss']
    ax = df[keys].plot(kind='line')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss(binary crossentropy)')
    plt.savefig(str(loss_path))

    keys = ['binary_accuracy', 'val_binary_accuracy']
    ax = df[keys].plot(kind='line')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    plt.savefig(str(acc_path))


if __name__ == '__main__':
    log, name = argv[1], argv[2]
    plot_svg(log, name)
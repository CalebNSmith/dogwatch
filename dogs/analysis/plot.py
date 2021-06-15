import matplotlib.pyplot as plt
import sys
import os
import shutil
import pandas
import glob
from collections import namedtuple

LOG_DIR = os.getenv('LOG_DIR')
PLOTS_DIR = os.getenv('DOGS_DIR') + '/plots/'
OLD_PLOTS_DIR = os.getenv('HOME') + '/archive/old_plots/'

def get_most_recent_logs(number=1):
    list_of_files = glob.glob(LOG_DIR + '*.log')
    latest = sorted(list_of_files, key=os.path.getctime, reverse=True)[:number]
    return latest

def logfile_to_tuples(log_file):
    tf_results = []
    with open(log_file) as f:
        # parse out keys from first line
        keys = f.readline().strip()
        Epoch = namedtuple('Epoch', keys.split(','))
        lines = f.readlines()
        for l in lines[1::]:
            l = l.strip()
            l = l.split(',')
            epoch = Epoch(*(tuple(l)))
            tf_results.append(epoch)
    return tf_results

def clean_out_plots():
    old_plots = glob.glob(PLOTS_DIR + '*.png')
    for op in old_plots:
        try:
            shutil.move(op, OLD_PLOTS_DIR)
        except:
            os.remove(op)

def create_plots(plot_name, tf_results):
    x = []
    loss = []
    cat_acc = []
    val_loss = []
    val_cat_acc = []

    epoch = 1
    for e in tf_results:
        x.append(epoch)
        loss.append(e.loss)
        cat_acc.append(e.categorical_accuracy)
        val_loss.append(e.val_loss)
        val_cat_acc.append(e.val_categorical_accuracy)
        epoch += 1

    fig, axs = plt.subplots(4, sharex=False, sharey=False, figsize=(6, 12))
    fig.tight_layout(h_pad=4)
    axs[0].plot(x, loss, 'tab:orange')
    axs[0].set_title('Loss')

    axs[1].plot(x, cat_acc, 'tab:blue')
    axs[1].set_title('Cat Acc')

    axs[2].plot(x, val_loss, 'tab:red')
    axs[2].set_title('Val Loss')

    axs[3].plot(x, val_cat_acc, 'tab:green')
    axs[3].set_title('Val Acc')

    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    plt.savefig(fname=plot_name, bbox_inches='tight', pad_inches=0.1, dpi=150)

def plot_logs(number=2):
    clean_out_plots()
    log_files = get_most_recent_logs()
    for lf in log_files:
        tf_results = logfile_to_tuples(lf)
        plot_name = PLOTS_DIR + lf.split('/')[-1].strip('.log') + '.png'
        print(plot_name)
        create_plots(plot_name, tf_results)

if __name__ == '__main__':
    plot_logs()

# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: visualization.py

@time: 2019/2/23 15:30

@desc:

"""

import matplotlib.pyplot as plt
import numpy as np
import itertools


# refer to https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, save_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification clases such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        normalize_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(normalize_cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if save_path is not None:
        plt.savefig(save_path)
        print('Logging Info - Confusion Matrix has save to', save_path)
    else:
        plt.show()


def plot_sequence_length(sequences, bins=50, title='sequence length', save_path=None):
    plt.title(title)
    plt.hist([len(seq) for seq in sequences], bins=bins)
    if save_path is not None:
        plt.savefig(save_path)
        print('Logging Info - Sequence Length has save to', save_path)
    else:
        plt.show()


def plot_line_chart(x_datas, y_datas, markers, labels, title, xlabel, ylabel, save_path=None, show=True):
    plt.grid(True)
    plt.ylim((0.45, 0.95))
    plt.yticks(np.arange(0.5, 1.0, 0.05))
    plt.xticks(np.arange(1, 9, 1.0))
    for x_data, y_data, marker, label in zip(x_datas, y_datas, markers, labels):
        plt.plot(x_data, y_data, marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels, loc=1)

    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        print('Logging Info - Sequence Length has save to', save_path)


if __name__ == '__main__':
    # plot_confusion_matrix(cm=np.array([[1098, 1934, 807],
    #                                    [604, 4392, 6233],
    #                                    [162, 2362, 31760]]),
    #                       normalize=True,
    #                       target_names=['high', 'medium', 'low'],
    #                       save_path='../img/cm.png',
    #                       title="Confusion Matrix")

    n_gram = [1, 2, 3, 4, 5, 6, 7, 8]
    s_svm_word = [0.8385, 0.7075, 0.5515, 0.515, 0.5035, 0.5015, 0.5005, 0.5]
    s_svm_char = [0.8115, 0.862, 0.876, 0.853, 0.816, 0.7905, 0.7295, 0.6505]
    s_lr_word = [0.859, 0.706, 0.552, 0.515, 0.5045, 0.5015, 0.5005, 0.5]
    s_lr_char = [0.8185, 0.872, 0.879, 0.8505, 0.824, 0.7805, 0.727, 0.65]
    s_mnb_word = [0.8785, 0.6575, 0.54, 0.51, 0.5035, 0.5015, 0.5005, 0.5005]
    s_mnb_char = [0.8225, 0.8935, 0.9015, 0.8835, 0.8455,  0.7755, 0.6795, 0.608]
    t_svm_word = [0.846, 0.7085, 0.5515, 0.515, 0.5035, 0.5015, 0.5005, 0.5]
    t_svm_char = [0.837, 0.884, 0.8845, 0.86, 0.823, 0.793, 0.73, 0.6505]
    t_lr_word = [0.8635, 0.707, 0.552, 0.515, 0.5045, 0.5015, 0.5005, 0.5]
    t_lr_char = [0.8455, 0.889, 0.884, 0.857, 0.8315, 0.7835, 0.7275, 0.65]
    t_mnb_word = [0.886, 0.658, 0.54, 0.51, 0.5035, 0.5015, 0.5005, 0.5005]
    t_mnb_char = [0.848, 0.91, 0.915, 0.891, 0.8495, 0.779, 0.68, 0.608]
    mark = ['ro-', 'bo-']
    labels = ['word', 'char']

    fig = plt.figure(figsize=(11, 5.5))
    plt.subplot(121)
    plot_line_chart([n_gram]*6, [s_svm_word, s_svm_char, s_lr_word, s_lr_char, s_mnb_word, s_mnb_char],
                    ['go-', 'go:', 'ro-', 'ro:', 'bo-', 'bo:'],
                    ['svm_word', 'svm_char', 'lr_word', 'lr_char', 'mnb_word', 'mnb_char'],
                    'Simplified version data set', 'N-gram', 'Accuracy', save_path=None, show=False)
    plt.subplot(122)
    plot_line_chart([n_gram] * 6, [t_svm_word, t_svm_char, t_lr_word, t_lr_char, t_mnb_word, t_mnb_char],
                    ['go-', 'go:', 'ro-', 'ro:', 'bo-', 'bo:'],
                    ['svm_word', 'svm_char', 'lr_word', 'lr_char', 'mnb_word', 'mnb_char'],
                    'Traditional version data set', 'N-gram', 'Accuracy', save_path=None, show=False)
    plt.savefig('../img/single_n_gram_1_2.png', bbox_inches='tight', dpi=200)
    # fig = plt.figure(figsize=(9, 6))
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.subplot(231)
    # plot_line_chart([n_gram, n_gram], [s_svm_word, s_svm_char], mark, labels, 'SVM', 'N-gram', 'Accuracy',
    #                 save_path=None, show=False)
    # plt.subplot(232)
    # plot_line_chart([n_gram, n_gram], [s_lr_word, s_lr_char], mark, labels, 'LR', 'N-gram', '',
    #                 save_path=None, show=False)
    # plt.subplot(233)
    # plot_line_chart([n_gram, n_gram], [s_mnb_word, s_mnb_char], mark, labels, 'MNB', 'N-gram', '',
    #                 save_path=None, show=False)
    # plt.subplot(234)
    # plot_line_chart([n_gram, n_gram], [t_svm_word, t_svm_char], mark, labels, '', 'N-gram', 'Accuracy',
    #                 save_path=None, show=False)
    # plt.subplot(235)
    # plot_line_chart([n_gram, n_gram], [t_lr_word, t_lr_char], mark, labels, '', 'N-gram', '',
    #                 save_path=None, show=False)
    # plt.subplot(236)
    # plot_line_chart([n_gram, n_gram], [t_mnb_word, t_mnb_char], mark, labels, '', 'N-gram', '',
    #                 save_path=None, show=False)
    # plt.savefig('../img/single_n_gram_3_3.png', bbox_inches='tight')



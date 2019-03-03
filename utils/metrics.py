# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: metrics.py

@time: 2019/2/9 15:24

@desc:

"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def eval_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def eval_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def eval_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def eval_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def eval_all(y_true, y_pred):
    return eval_acc(y_true, y_pred), eval_f1(y_true, y_pred), eval_precision(y_true, y_pred), eval_recall(y_true, y_pred)


def return_error_index(y_true, y_pred):
    return np.nonzero((y_true == y_pred) == 0)[0]


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)




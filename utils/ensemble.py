# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: ensemble_models.py

@time: 2019/2/19 23:18

@desc:

"""

import numpy as np
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


# ensembling multiple models by averaging their output probabilities
def mean_ensemble(model_pred_probas, threshold=0.5):
    model_pred_probas = np.column_stack(model_pred_probas)
    ensemble_pred_prob = np.mean(model_pred_probas, axis=-1)
    ensemble_pred_class = [1 if pred_proba >= threshold else 0 for pred_proba in ensemble_pred_prob]
    return np.array(ensemble_pred_class)


# ensembling multiple models by taking the highest probability for a certain label
def max_ensemble(model_pred_probas, threshold=0.5):
    nb_model = len(model_pred_probas)
    nb_sample = model_pred_probas[0].shape[0]

    model_pred_max_probas = []
    for model_pred_proba in model_pred_probas:
        model_pred_full_proba = np.column_stack((1-model_pred_proba, model_pred_proba))
        model_pred_max_probas.append(np.max(model_pred_full_proba, axis=-1))

    model_pred_max_probas = np.column_stack(model_pred_max_probas)
    assert model_pred_max_probas.shape == (nb_sample, nb_model)

    model_pred_max_model_index = np.argmax(model_pred_max_probas, axis=-1)
    ensemble_pred_proba = []
    for i in range(nb_sample):
        model_index = model_pred_max_model_index[i]
        ensemble_pred_proba.append(model_pred_probas[model_index][i])

    ensemble_pred_class = [1 if pred_proba >= threshold else 0 for pred_proba in ensemble_pred_proba]
    return ensemble_pred_class


# majority-vote ensembling, if all classifier disagree(tie), return label by the classifier with index fallback
def vote_ensemble(model_pred_classes, fallback=0):
    nb_model = len(model_pred_classes)
    ensemble_pred_class = []
    for sample_pred in zip(*model_pred_classes):    # group by sample
        class_label, class_count = Counter(sample_pred).most_common(1)[0]   # majority class
        if class_count > nb_model / 2:  # more than half
            ensemble_pred_class.append(class_label)
        else:
            ensemble_pred_class.append(sample_pred[fallback])

    return np.array(ensemble_pred_class)

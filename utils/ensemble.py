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


# ensembling multiple models by averaging their output probabilities
def mean_ensemble(model_pred_probas, threshold=0.5):
    model_pred_probas = [np.expand_dims(pred_proba, -1) for pred_proba in model_pred_probas]
    ensemble_pred_prob = np.mean(np.concatenate(model_pred_probas, axis=-1), axis=-1)
    ensemble_pred_class = [1 if pred_proba >= threshold else 0 for pred_proba in ensemble_pred_prob]
    return ensemble_pred_class


# ensembling multiple models by taking the highest probability
def max_ensemble(model_pred_probas, threshold=0.5):
    model_pred_probas = [np.expand_dims(pred_proba, -1) for pred_proba in model_pred_probas]
    ensemble_pred_prob = np.max(np.concatenate(model_pred_probas, axis=-1), axis=-1)
    ensemble_pred_class = [1 if pred_proba >= threshold else 0 for pred_proba in ensemble_pred_prob]
    return ensemble_pred_class


# majority-vote ensembling, if all classifier disagree, return label by the classifier with index fallback
def vote_ensemble(model_pred_classes, fallback=0):
    ensemble_pred_class = []
    for sample_pred in zip(*model_pred_classes):    # group by sample
        class_label, class_count = Counter(sample_pred).most_common(1)[0]   # majority class
        if class_count > 2:
            ensemble_pred_class.append(class_label)
        else:
            ensemble_pred_class.append(sample_pred[fallback])
    return np.array(ensemble_pred_class)
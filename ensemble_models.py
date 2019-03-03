# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: ensemble_models.py

@time: 2019/2/19 13:16

@desc:

"""


import os
import time
from os import path
import numpy as np
from config import ModelConfig, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, LOG_DIR, PERFORMANCE_LOG_TEMPLATE, \
    VARIATIONS
from train import get_optimizer
from models.keras_bilstm_model import BiLSTM
from models.keras_cnnrnn_model import CNNRNN
from models.keras_dcnn_model import DCNN
from models.keras_dpcnn_model import DPCNN
from models.keras_han_model import HAN
from models.keras_multi_text_cnn_model import MultiTextCNN
from models.keras_rcnn_model import RCNN
from models.keras_rnncnn_model import RNNCNN
from models.keras_text_cnn_model import TextCNN
from models.keras_vdcnn_model import VDCNN
from models.sklearn_base_model import SVMModel, LRModel, SGDModel, GaussianNBModel, MultinomialNBModel, \
    BernoulliNBModel, RandomForestModel, GBDTModel, XGBoostModel, LDAModel
from utils.io import format_filename, write_log
from utils.data_loader import load_ngram_data, load_processed_data, load_processed_text_data
from utils.metrics import eval_all
from utils.ensemble import mean_ensemble, max_ensemble, vote_ensemble


def train_ensemble_model(ensemble_models, model_name, variation, dev_data, train_data=None, binary_threshold=0.5,
                         checkpoint_dir=None, overwrite=False, log_error=False, **kwargs):
    config = ModelConfig()
    config.binary_threshold = binary_threshold
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
        if not path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
    config.exp_name = '{}_{}_ensemble_with_{}'.format(variation, model_name, ensemble_models)
    train_log = {'exp_name': config.exp_name, 'binary_threshold': binary_threshold}
    print('Logging Info - Ensemble Experiment: ', config.exp_name)
    if model_name == 'svm':
        model = SVMModel(config, **kwargs)
    elif model_name == 'lr':
        model = LRModel(config, **kwargs)
    elif model_name == 'sgd':
        model = SGDModel(config, **kwargs)
    elif model_name == 'gnb':
        model = GaussianNBModel(config, **kwargs)
    elif model_name == 'mnb':
        model = MultinomialNBModel(config, **kwargs)
    elif model_name == 'bnb':
        model = BernoulliNBModel(config, **kwargs)
    elif model_name == 'rf':
        model = RandomForestModel(config, **kwargs)
    elif model_name == 'gbdt':
        model = GBDTModel(config, **kwargs)
    elif model_name == 'xgboost':
        model = XGBoostModel(config, **kwargs)
    elif model_name == 'lda':
        model = LDAModel(config, **kwargs)
    else:
        raise ValueError('Model Name Not Understood : {}'.format(model_name))

    model_save_path = path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if train_data is not None and (not path.exists(model_save_path) or overwrite):
        model.train(train_data)

    model.load_best_model()
    print('Logging Info - Evaluate over valid data:')
    valid_acc, valid_f1, valid_p, valid_r = model.evaluate(dev_data)
    train_log['valid_acc'] = valid_acc
    train_log['valid_f1'] = valid_f1
    train_log['valid_p'] = valid_p
    train_log['valid_r'] = valid_r
    train_log['time_stamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if log_error:
        error_indexes, error_pred_probas = model.error_analyze(dev_data)
        dev_text_input = load_processed_text_data(variation, 'dev')
        for error_index, error_pred_prob in zip(error_indexes, error_pred_probas):
            train_log['error_%d' % error_index] = '{},{},{},{}'.format(error_index,
                                                                       dev_text_input['sentence'][error_index],
                                                                       dev_text_input['label'][error_index],
                                                                       error_pred_prob)

    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation), log=train_log, mode='a')
    return valid_acc, valid_f1, valid_p, valid_r


def predict_dl_model(data_type, variation, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
                     optimizer_type, model_name, checkpoint_dir=None, return_proba=True, **kwargs):
    config = ModelConfig()
    config.variation = variation
    config.input_level = input_level
    if '_aug' in variation:
        config.max_len = {'word': config.aug_word_max_len, 'char': config.aug_char_max_len}
    config.word_embed_type = word_embed_type
    config.word_embed_trainable = word_embed_trainable
    config.word_embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE,
                                                     variation=variation, type=word_embed_type))
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    config.exp_name = '{}_{}_{}_{}_{}'.format(variation, model_name, input_level, word_embed_type,
                                              'tune' if word_embed_trainable else 'fix')

    print('Logging Info - Experiment: ', config.exp_name)
    if model_name == 'bilstm':
        model = BiLSTM(config, **kwargs)
    elif model_name == 'cnnrnn':
        model = CNNRNN(config, **kwargs)
    elif model_name == 'dcnn':
        model = DCNN(config, **kwargs)
    elif model_name == 'dpcnn':
        model = DPCNN(config, **kwargs)
    elif model_name == 'han':
        model = HAN(config, **kwargs)
    elif model_name == 'multicnn':
        model = MultiTextCNN(config, **kwargs)
    elif model_name == 'rcnn':
        model = RCNN(config, **kwargs)
    elif model_name == 'rnncnn':
        model = RNNCNN(config, **kwargs)
    elif model_name == 'cnn':
        model = TextCNN(config, **kwargs)
    elif model_name == 'vdcnn':
        model = VDCNN(config, **kwargs)
    else:
        raise ValueError('Model Name Not Understood : {}'.format(model_name))

    model_save_path = path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not path.exists(model_save_path):
        raise FileNotFoundError('Model Not Found: {}'.format(model_save_path))
    # load the best model
    model.load_best_model()

    data = load_processed_data(variation, input_level, data_type)

    if return_proba:
        return model.predict_proba(data), config.exp_name
    else:
        return model.predict(data), config.exp_name


def predict_ml_model(data_type, model_name, variation, vectorizer_type, level, ngram_range, checkpoint_dir=None,
                     return_proba=True, **kwargs):
    config = ModelConfig()
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    config.exp_name = '{}_{}_{}_{}_{}'.format(variation, model_name, vectorizer_type, level, ngram_range)
    print('Logging Info - Experiment: ', config.exp_name)
    if model_name == 'svm':
        model = SVMModel(config, **kwargs)
    elif model_name == 'lr':
        model = LRModel(config, **kwargs)
    elif model_name == 'sgd':
        model = SGDModel(config, **kwargs)
    elif model_name == 'gnb':
        model = GaussianNBModel(config, **kwargs)
    elif model_name == 'mnb':
        model = MultinomialNBModel(config, **kwargs)
    elif model_name == 'bnb':
        model = BernoulliNBModel(config, **kwargs)
    elif model_name == 'rf':
        model = RandomForestModel(config, **kwargs)
    elif model_name == 'gbdt':
        model = GBDTModel(config, **kwargs)
    elif model_name == 'xgboost':
        model = XGBoostModel(config, **kwargs)
    else:
        raise ValueError('Model Name Not Understood : {}'.format(model_name))

    model_save_path = path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not path.exists(model_save_path):
        raise FileNotFoundError('Model Not Found: {}'.format(model_save_path))

    model.load_best_model()
    data = load_ngram_data(variation, vectorizer_type, level, ngram_range, data_type)
    if return_proba:
        return model.predict_proba(data), config.exp_name
    else:
        return model.predict(data), config.exp_name


if __name__ == '__main__':
    retrain = True
    for variation in VARIATIONS:
        # prepare models' output probability as input for ensemble model
        train_model_pred_probas = []
        train_label = load_processed_data(variation, 'word', 'train')['label']
        dev_model_pred_probas = []
        dev_label = load_processed_data(variation, 'word', 'dev')['label']

        ensemble_log = {'ensmeble_models': []}
        ensemble_models = []

        for dl_model_name in ['cnn', 'cnnrnn', 'dcnn', 'dpcnn', 'han', 'rcnn', 'rnncnn', 'bilstm']:
            if retrain:
                train_pred_proba, exp_name = predict_dl_model('train', variation, 'word', 'w2v_data', True, 64,
                                                              0.001, 'adam', dl_model_name, return_proba=True)
                train_model_pred_probas.append(train_pred_proba)
            dev_pred_proba, exp_name = predict_dl_model('dev', variation, 'word', 'w2v_data', True, 64, 0.001,
                                                        'adam', dl_model_name, return_proba=True)
            dev_model_pred_probas.append(dev_pred_proba)
            ensemble_models.append(exp_name)
            ensemble_log['ensmeble_models'].append(exp_name)

        for ml_model_name in []:
            if retrain:
                train_pred_proba, exp_name = predict_ml_model('train', ml_model_name, variation, 'binary',
                                                              'char', (2, 3), return_proba=True)
                train_pred_proba = train_pred_proba[:, 1]
                train_model_pred_probas.append(train_pred_proba)
            dev_pred_proba, exp_name = predict_ml_model('dev', ml_model_name, variation, 'binary', 'char', (2, 3),
                                                        return_proba=True)
            dev_pred_proba = dev_pred_proba[:, 1]
            dev_model_pred_probas.append(dev_pred_proba)
            ensemble_models.append(exp_name)
            ensemble_log['ensmeble_models'].append(exp_name)

        if retrain:
            train_model_pred_probas = np.column_stack(train_model_pred_probas)
            train_ensemble_input = {'sentence': train_model_pred_probas, 'label': train_label}
        else:
            train_ensemble_input = None
        dev_model_pred_probas = np.column_stack(dev_model_pred_probas)
        dev_ensemble_input = {'sentence': dev_model_pred_probas, 'label': dev_label}

        for binary_threshold in np.arange(0.4, 0.62, 0.02):
            ensemble_log['binary_threshold'] = binary_threshold
            for model_name in ['svm', 'lr', 'gnb', 'mnb', 'bnb', 'rf', 'xgboost', 'gbdt', 'lda']:
                performance = train_ensemble_model('all_dl_model', model_name, variation, dev_ensemble_input,
                                                   train_ensemble_input, binary_threshold=binary_threshold)
                print('Logging Info - {} - meta-classifier ensembling: (acc, f1, p, r):{}'.format(variation,
                                                                                                  performance))
                ensemble_log['%s_ensemble' % model_name] = performance
            ensemble_log['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation + '_ensemble'),
                  ensemble_log, mode='a')

    for binary_threshold in np.arange(0.4, 0.62, 0.02):
        for variation in VARIATIONS:
            model_pred_probas = []
            model_pred_classes = []

            dev_data = load_processed_data(variation, 'word', 'dev')
            ensemble_log = {'ensmeble_models': [], 'binary_threshold': binary_threshold}
            for dl_model_name in ['cnn', 'cnnrnn', 'dcnn', 'dpcnn', 'han', 'rcnn', 'rnncnn', 'bilstm']:
                pred_proba, exp_name = predict_dl_model('dev', variation, 'word', 'w2v_data', True, 64, 0.001, 'adam',
                                                        dl_model_name, return_proba=True)
                pred_class = np.array([1 if proba >= binary_threshold else 0 for proba in pred_proba])
                model_pred_probas.append(pred_proba)
                model_pred_classes.append(pred_class)
                ensemble_log['ensmeble_models'].append(exp_name)

            for ml_model_name in []:
                pred_proba, exp_name = predict_ml_model('dev', ml_model_name, variation, 'binary', 'char', (2, 3),
                                                        return_proba=True)
                pred_proba = pred_proba[:, 1]
                pred_class = np.array([1 if proba >= binary_threshold else 0 for proba in pred_proba])
                model_pred_probas.append(pred_proba)
                model_pred_classes.append(pred_class)
                ensemble_log['ensmeble_models'].append(exp_name)

            mean_pred_class = mean_ensemble(model_pred_probas, binary_threshold)
            mean_performance = eval_all(dev_data['label'], mean_pred_class)
            ensemble_log['mean_ensemble'] = mean_performance
            print('Logging Info - {} - mean ensembling: (acc, f1, p, r):{}'.format(variation, mean_performance))

            max_pred_class = max_ensemble(model_pred_probas, binary_threshold)
            max_performance = eval_all(dev_data['label'], max_pred_class)
            ensemble_log['max_ensemble'] = max_performance
            print('Logging Info - {} - max ensembling: (acc, f1, p, r):{}'.format(variation, max_performance))

            vote_pred_class = vote_ensemble(model_pred_classes, fallback=-1)
            vote_performance = eval_all(dev_data['label'], vote_pred_class)
            ensemble_log['vote_ensemble'] = vote_performance
            print('Logging Info - {} - majority vote ensembling: (acc, f1, p, r):{}'.format(variation, vote_performance))

            ensemble_log['time_stamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation+'_ensemble'), ensemble_log,
                      mode='a')



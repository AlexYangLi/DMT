# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: ensemble_models.py

@time: 2019/2/19 13:16

@desc:

"""


from os import path
import numpy as np

from config import ModelConfig, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, LOG_DIR, PERFORMANCE_LOG_TEMPLATE
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
    BernoulliNBModel, RandomForestModel, GBDTModel, XGBoostModel
from utils.io import format_filename, write_log
from utils.data_loader import load_ngram_data, load_processed_data
from utils.metrics import eval_all
from utils.ensemble import mean_ensemble, max_ensemble, vote_ensemble


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
    binary_threshold = ModelConfig().binary_threshold

    for variation in ['simplified', 'traditional']:
        model_pred_probas = []
        model_pred_classes = []

        data = load_processed_data(variation, 'word', 'dev')
        ensemble_log = {'ensmeble_models': []}
        for model_name in ['rcnn']:

            pred_proba, exp_name = predict_dl_model('dev', variation, 'word', 'w2v_data', True, 64, 0.001, 'adam',
                                                    model_name)
            pred_class = np.array([1 if proba >= 0.5 else 0 for proba in pred_proba])
            model_pred_probas.append(pred_proba)
            model_pred_classes.append(pred_class)
            ensemble_log['ensmeble_models'].append(exp_name)

        mnb_proba, exp_name = predict_ml_model('dev', 'mnb', variation, 'binary', 'char', (2, 3))
        mnb_proba = mnb_proba[:, 1]
        mnb_class = np.array([1 if proba >= 0.5 else 0 for proba in mnb_proba])
        model_pred_probas.append(mnb_proba)
        model_pred_classes.append(mnb_class)
        ensemble_log['ensmeble_models'].append(exp_name)

        mean_pred_class = mean_ensemble(model_pred_probas, binary_threshold)
        mean_performance = eval_all(data['label'], mean_pred_class)
        ensemble_log['mean_ensemble'] = mean_performance
        print('Logging Info - {} - mean ensembling: (acc, f1, p, r):{}'.format(variation, mean_performance))

        max_pred_class = max_ensemble(model_pred_probas, binary_threshold)
        max_performance = eval_all(data['label'], max_pred_class)
        ensemble_log['max_ensemble'] = max_performance
        print('Logging Info - {} - max ensembling: (acc, f1, p, r):{}'.format(variation, max_performance))

        vote_pred_class = vote_ensemble(model_pred_classes)
        vote_performance = eval_all(data['label'], max_pred_class)
        ensemble_log['vote_ensemble'] = vote_performance
        print('Logging Info - {} - majority vote ensembling: (acc, f1, p, r):{}'.format(variation, vote_performance))

        write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation+'_ensemble'), ensemble_log,
                  mode='a')









# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2019/2/9 13:31

@desc:

"""

import os
from os import path
import time
import numpy as np
from keras import optimizers

from config import ModelConfig, LOG_DIR, PERFORMANCE_LOG_TEMPLATE, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, \
    TRAIN_NGRAM_DATA_TEMPLATE, DEV_NGRAM_DATA_TEMPLATE
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

from utils.data_loader import load_processed_data, load_ngram_data
from utils.io import format_filename, write_log

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


# train deep learning based model
def train_dl_model(variation, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
                   optimizer_type, model_name, checkpoint_dir=None, **kwargs):
    config = ModelConfig()
    config.variation = variation
    config.input_level = input_level
    config.word_embed_type = word_embed_type
    config.word_embed_trainable = word_embed_trainable
    config.word_embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE,
                                                     variation=variation, type=word_embed_type))
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
    config.exp_name = '{}_{}_{}_{}_{}'.format(variation, model_name, input_level, word_embed_type,
                                              'tune' if word_embed_trainable else 'fix')

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'learning_rate': learning_rate}

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

    train_input = load_processed_data(variation, input_level, 'train')
    dev_input = load_processed_data(variation, input_level, 'dev')

    model_save_path = path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not path.exists(model_save_path):
        start_time = time.time()
        model.train(train_input, dev_input)
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # load the best model
    model.load_best_model()

    print('Logging Info - Evaluate over valid data:')
    valid_acc, valid_f1 = model.evaluate(dev_input)
    train_log['valid_acc'] = valid_acc
    train_log['valid_f1'] = valid_f1
    train_log['time_stamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation), log=train_log, mode='a')
    return valid_acc, valid_f1


# train machine learning based model
def train_ml_model(model_name, variation, vectorizer_type, level, ngram_range, checkpoint_dir=None, **kwargs):
    config = ModelConfig()
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
    config.exp_name = '{}_{}_{}_{}_{}'.format(variation, model_name, vectorizer_type, level, ngram_range)
    train_log = {'exp_name': config.exp_name}
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

    train_input = load_ngram_data(variation, vectorizer_type, level, ngram_range, 'train')
    dev_input = load_ngram_data(variation, vectorizer_type, level, ngram_range, 'dev')

    model_save_path = path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not path.exists(model_save_path):
        model.train(train_input)

    model.load_best_model()
    print('Logging Info - Evaluate over valid data:')
    valid_acc, valid_f1, valid_p, valid_r = model.evaluate(dev_input)
    train_log['valid_acc'] = valid_acc
    train_log['valid_f1'] = valid_f1
    train_log['valid_p'] = valid_p
    train_log['valid_r'] = valid_r
    train_log['time_stamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation), log=train_log, mode='a')
    return valid_acc, valid_f1, valid_p, valid_r


if __name__ == '__main__':
    for model_name in ['lr', 'svm', 'sgd', 'mnb', 'bnb', 'rf', 'gbdt', 'xgboost']:
        for variation in ['simplified', 'traditional']:
            for vectorizer_type in ['binary', 'tf', 'tfidf']:
                train_ml_model(model_name, variation, vectorizer_type, 'char', (3, 3))
                train_ml_model(model_name, variation, vectorizer_type, 'char', (2, 3))
                train_ml_model(model_name, variation, vectorizer_type, 'word', (1, 1))
                train_ml_model(model_name, variation, vectorizer_type, ['char', 'word'], [(2, 3), (1, 1)])

    # from collections import defaultdict
    # simp_acc, simp_f1, trad_acc, trad_f1 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    # for i in range(10):
    #     checkpoint_dir = './ckpt/%d' % i
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'cnnrnn', checkpoint_dir)
    #     simp_acc['cnnrnn'].append(valid_acc)
    #     simp_f1['cnnrnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm', checkpoint_dir)
    #     simp_acc['bilstm'].append(valid_acc)
    #     simp_f1['bilstm'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn', checkpoint_dir)
    #     simp_acc['dcnn'].append(valid_acc)
    #     simp_f1['dcnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn', checkpoint_dir)
    #     simp_acc['dpcnn'].append(valid_acc)
    #     simp_f1['dpcnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'han', checkpoint_dir)
    #     simp_acc['han'].append(valid_acc)
    #     simp_f1['han'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn', checkpoint_dir)
    #     simp_acc['multicnn'].append(valid_acc)
    #     simp_f1['multicnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn', checkpoint_dir)
    #     simp_acc['rcnn'].append(valid_acc)
    #     simp_f1['rcnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn', checkpoint_dir)
    #     simp_acc['rnncnn'].append(valid_acc)
    #     simp_f1['rnncnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'cnn', checkpoint_dir)
    #     simp_acc['cnn'].append(valid_acc)
    #     simp_f1['cnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn', checkpoint_dir)
    #     simp_acc['vdcnn'].append(valid_acc)
    #     simp_f1['vdcnn'].append(valid_f1)
    #
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'cnnrnn', checkpoint_dir)
    #     trad_acc['cnnrnn'].append(valid_acc)
    #     trad_f1['cnnrnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm', checkpoint_dir)
    #     trad_acc['bilstm'].append(valid_acc)
    #     trad_f1['bilstm'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn', checkpoint_dir)
    #     trad_acc['dcnn'].append(valid_acc)
    #     trad_f1['dcnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn', checkpoint_dir)
    #     trad_acc['dpcnn'].append(valid_acc)
    #     trad_f1['dpcnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'han', checkpoint_dir)
    #     trad_acc['han'].append(valid_acc)
    #     trad_f1['han'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn', checkpoint_dir)
    #     trad_acc['multicnn'].append(valid_acc)
    #     trad_f1['multicnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn', checkpoint_dir)
    #     trad_acc['rcnn'].append(valid_acc)
    #     trad_f1['rcnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn', checkpoint_dir)
    #     trad_acc['rnncnn'].append(valid_acc)
    #     trad_f1['rnncnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'cnn', checkpoint_dir)
    #     trad_acc['cnn'].append(valid_acc)
    #     trad_f1['cnn'].append(valid_f1)
    #     valid_acc, valid_f1 = train_dl_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn', checkpoint_dir)
    #     trad_acc['vdcnn'].append(valid_acc)
    #     trad_f1['vdcnn'].append(valid_f1)
    #
    # for model in simp_acc:
    #     results = simp_acc[model]
    #     simp_acc[model] = [results, np.max(results), np.argmax(results), np.mean(results), np.std(results)]
    # for model in simp_f1:
    #     results = simp_f1[model]
    #     simp_f1[model] = [results, np.max(results), np.argmax(results), np.mean(results), np.std(results)]
    # for model in trad_acc:
    #     results = trad_acc[model]
    #     trad_acc[model] = [results, np.max(results), np.argmax(results), np.mean(results), np.std(results)]
    # for model in trad_f1:
    #     results = trad_f1[model]
    #     trad_f1[model] = [np.max(results), np.argmax(results), np.mean(results), np.std(results)]
    # write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation='all'), log=simp_acc, mode='a')
    # write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation='all'), log=simp_f1, mode='a')
    # write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation='all'), log=trad_acc, mode='a')
    # write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation='all'), log=trad_f1, mode='a')

    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    # train_dl_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')

    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'han')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'cnn')
    # train_dl_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn')
    #
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    # train_dl_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')

    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    # train_dl_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')
    #
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'han')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'cnn')
    # train_dl_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn')
    #
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    # train_dl_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')


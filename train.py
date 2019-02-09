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

from config import ModelConfig, LOG_DIR, PERFORMANCE_LOG_TEMPLATE, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE
from models.keras_bilstm_model import BiLSTM
from models.keras_dcnn_model import DCNN
from models.keras_dpcnn_model import DPCNN
from models.keras_han_model import HAN
from models.keras_multi_text_cnn_model import MultiTextCNN
from models.keras_rcnn_model import RCNN
from models.keras_rnncnn_model import RNNCNN
from models.keras_text_cnn_model import TextCNN
from models.keras_vdcnn_model import VDCNN
from utils.data_loader import load_processed_data
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


def train_model(variation, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer_type,
                model_name):
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
    config.exp_name = '{}_{}_{}_{}_{}'.format(variation, model_name, input_level, word_embed_type,
                                              'tune' if word_embed_trainable else 'fix')

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'learning_rate': learning_rate}

    print('Logging Info - Experiment: ', config.exp_name)
    if model_name == 'bilstm':
        model = BiLSTM(config)
    elif model_name == 'dcnn':
        model = DCNN(config)
    elif model_name == 'dpcnn':
        model = DPCNN(config)
    elif model_name == 'han':
        model = HAN(config)
    elif model_name == 'multicnn':
        model = MultiTextCNN(config)
    elif model_name == 'rcnn':
        model = RCNN(config)
    elif model_name == 'rnncnn':
        model = RNNCNN(config)
    elif model_name == 'cnn':
        model = TextCNN(config)
    elif model_name == 'vdcnn':
        model = VDCNN(config)
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
    valid_acc = model.evaluate(dev_input)
    train_log['valid_acc'] = valid_acc

    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, variation=variation), log=train_log, mode='a')


if __name__ == '__main__':
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'han')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'cnn')
    train_model('simplified', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn')

    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    train_model('simplified', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')

    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'han')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'cnn')
    train_model('simplified', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn')

    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    train_model('simplified', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')

    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm')
    train_model('traditionaltraditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'han')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'cnn')
    train_model('traditional', 'word', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn')

    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    train_model('traditional', 'word', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')

    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'bilstm')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dcnn')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'dpcnn')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'han')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'multicnn')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rcnn')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'rnncnn')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'cnn')
    train_model('traditional', 'char', 'w2v_data', True, 64, 0.001, 'adam', 'vdcnn')

    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'bilstm')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dcnn')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'dpcnn')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'han')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'multicnn')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rcnn')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'rnncnn')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'cnn')
    train_model('traditional', 'char', 'w2v_data', False, 64, 0.001, 'adam', 'vdcnn')
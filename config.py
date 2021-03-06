# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: config.py

@time: 2019/2/9 9:54

@desc:

"""

from os import path

from keras.optimizers import Adam

RAW_DATA_DIR = './raw_data'
PROCESSED_DATA_DIR = './data'
LOG_DIR = './log'
MODEL_SAVED_DIR = './ckpt'
IMG_DIR = './img'
PREDICT_DIR = './predict'

SIMP_DIR = path.join(RAW_DATA_DIR, 'TRAININGSET-DMT_SIMP-VARDIAL2019')
SIMP_TEST_DIR = path.join(RAW_DATA_DIR, 'dmt_simp_test')
SIMP_TRAIN_FILENAME = path.join(SIMP_DIR, 'train.txt')
SIMP_DEV_FILENAME = path.join(SIMP_DIR, 'dev.txt')
SIMP_TEST_FILENAME = path.join(SIMP_TEST_DIR, 'dmt-simp-test.txt')

TRAD_DIR = path.join(RAW_DATA_DIR, 'TRAININGSET-DMT_TRAD-VARDIAL2019')
TRAD_TEST_DIR = path.join(RAW_DATA_DIR, 'dmt_trad_test')
TRAD_TRAIN_FILENAME = path.join(TRAD_DIR, 'train.txt')
TRAD_DEV_FILENAME = path.join(TRAD_DIR, 'dev.txt')
TRAD_TEST_FILENAME = path.join(TRAD_TEST_DIR, 'dmt-trad-test.txt')

TRAIN_DATA_TEMPLATE = '{variation}_train.pkl'
DEV_DATA_TEMPLATE = '{variation}_dev.pkl'
TEST_DATA_TEMPLATE = '{variation}_test.pkl'

TRAIN_IDS_MATRIX_TEMPLATE = '{variation}_{level}_ids_train.pkl'
DEV_IDS_MATRIX_TEMPLATE = '{variation}_{level}_ids_dev.pkl'
TEST_IDS_MATRIX_TEMPLATE = '{variation}_{level}_ids_test.pkl'

EMBEDDING_MATRIX_TEMPLATE = '{variation}_{type}_embeddings.npy'
TOKENIZER_TEMPLATE = '{variation}_{level}_tokenizer.pkl'
VOCABULARY_TEMPLATE = '{variation}_{level}_vocab.pkl'

TRAIN_NGRAM_DATA_TEMPLATE = '{variation}_{type}_{level}_{ngram_range}_train.pkl'
DEV_NGRAM_DATA_TEMPLATE = '{variation}_{type}_{level}_{ngram_range}_dev.pkl'
TEST_NGRAM_DATA_TEMPLATE = '{variation}_{type}_{level}_{ngram_range}_test.pkl'

VECTORIZER_TEMPLATE = '{variation}_{type}_{level}_{ngram_range}_vectorizer.pkl'

ANALYSIS_LOG_TEMPLATE = '{variation}_analysis.log'
PERFORMANCE_LOG_TEMPLATE = '{variation}_performance.log'

EXTERNAL_WORD_VECTORS_DIR = path.join(RAW_DATA_DIR, 'word_embeddings/')
EXTERNAL_WORD_VECTORS_FILENAME = {}

LABELS = {'T': 0, 'M': 1}
ID_TO_LABELS = {0: 'T', 1: 'M'}
VARIATIONS = ['simplified', 'traditional']


class ModelConfig(object):
    def __init__(self):
        # input configuration
        self.data_augment = False
        self.variation = 'simplified'
        self.input_level = 'word'
        self.word_max_len = 66
        self.char_max_len = 155
        self.aug_word_max_len = 44
        self.aug_char_max_len = 103
        self.max_len = {'word': self.word_max_len, 'char': self.char_max_len}
        self.word_embed_type = 'w2v'
        self.word_embed_dim = 300
        self.word_embed_trainable = False

        # output configuration
        self.n_class = 2

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.rnn_units = 300
        self.dense_units = 128

        # model training configuration
        self.batch_size = 32
        self.n_epoch = 50
        self.learning_rate = 0.001
        self.optimizer = Adam(self.learning_rate)
        self.dropout = 0.5
        self.l2_reg = 0.001
        self.loss_function = 'binary_crossentropy'
        self.binary_threshold = 0.5

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_acc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_acc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

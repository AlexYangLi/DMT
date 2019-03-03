# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: preprocess.py

@time: 2019/2/9 13:31

@desc:

"""

import os
from os import path
import itertools
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from config import SIMP_TRAIN_FILENAME, SIMP_DEV_FILENAME, TRAD_TRAIN_FILENAME, TRAD_DEV_FILENAME, TRAIN_DATA_TEMPLATE, \
    DEV_DATA_TEMPLATE, TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, EMBEDDING_MATRIX_TEMPLATE, \
    TOKENIZER_TEMPLATE, VOCABULARY_TEMPLATE, ANALYSIS_LOG_TEMPLATE
from config import PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, IMG_DIR
from config import LABELS, VARIATIONS
from config import ModelConfig
from utils.data_loader import read_raw_data
from utils.analysis import analyze_len_distribution, analyze_class_distribution
from utils.embeddings import train_w2v, train_glove, train_fasttext
from utils.io import pickle_dump, format_filename, write_log

# Todo: add prepare_ngram_feature.py and data_augment.py


def get_sentence_label(data):
    labels = data['label'].map(LABELS).tolist()
    sentences = data['sentence'].tolist()
    return {'sentence': sentences, 'label': labels}


def load_data():
    """Load raw data into train/dev DataFrames"""
    data_simp_train = read_raw_data(SIMP_TRAIN_FILENAME, set_variation='simplified')
    data_simp_dev = read_raw_data(SIMP_DEV_FILENAME, set_variation='simplified')
    print('Logging Info - Simplified: train - {}, dev - {}'.format(data_simp_train.shape, data_simp_dev.shape))

    data_trad_train = read_raw_data(TRAD_TRAIN_FILENAME, set_variation='traditional')
    data_trad_dev = read_raw_data(TRAD_DEV_FILENAME, set_variation='traditional')
    print('Logging Info - Traditional: train - {}, dev - {}'.format(data_trad_train.shape, data_trad_dev.shape))

    # data_all_train = pd.concat([data_simp_train, data_trad_train])
    # data_all_dev = pd.concat([data_simp_dev, data_trad_dev])
    # data_all_train['variation'] = 'all'
    # data_all_dev['variation'] = 'all'

    # concatenate all data together
    data_train = pd.concat([data_simp_train, data_trad_train])
    data_dev = pd.concat([data_simp_dev, data_trad_dev])

    data_train.set_index('variation', inplace=True)
    data_dev.set_index('variation', inplace=True)

    return data_train, data_dev


def create_token_ids_matrix(tokenizer, sequences, max_len=None):
    tokens_ids = tokenizer.texts_to_sequences(sequences)

    # there might be zero len sequences - fix it by putting a random token there (or id 1 in the worst case)
    tokens_ids_flattened = list(itertools.chain.from_iterable(tokens_ids))
    max_id = max(tokens_ids_flattened) if len(tokens_ids_flattened) > 0 else -1

    for i in range(len(tokens_ids)):
        if len(tokens_ids[i]) == 0:
            id_to_put = np.random.randint(1, max_id) if max_id != -1 else 1
            tokens_ids[i].append(id_to_put)

    print('Logging Info - pad sequence with max_len = %d', max_len)
    tokens_ids = pad_sequences(tokens_ids, maxlen=max_len)
    return tokens_ids


def create_data_matrices(tokenizer, data, n_class, one_hot, max_len=None):
    sentence = create_token_ids_matrix(tokenizer, data['sentence'], max_len)
    if one_hot:
        label = to_categorical(data['label'], n_class)
    else:
        label = np.array(data['label'])

    m_data = {
        'sentence': sentence,
        'label': label,
    }
    return m_data


def process_data():
    config = ModelConfig()

    # create dir
    if not path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    if not path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # load datasets
    data_train, data_dev = load_data()
    print('Logging Info - Data: train - {}, dev - {}'.format(data_train.shape, data_dev.shape))

    for variation in VARIATIONS:
        if variation not in data_train.index:
            continue

        analyze_result = {}
        variation_train = data_train.loc[variation]
        variation_dev = data_dev.loc[variation]

        print('Logging Info - Variation: {}, train - {}, dev - {}'.format(variation, variation_train.shape,
                                                                          variation_dev.shape))
        analyze_result.update({'train_set': len(variation_train), 'dev_set': len(variation_train)})

        variation_train_data = get_sentence_label(variation_train)
        variation_dev_data = get_sentence_label(variation_dev)

        # class distribution analysis
        train_label_distribution = analyze_class_distribution(variation_train_data['label'])
        analyze_result.update(
            dict(('train_cls_{}'.format(cls), percent) for cls, percent in train_label_distribution.items()))
        dev_label_distribution = analyze_class_distribution(variation_dev_data['label'])
        analyze_result.update(
            dict(('dev_cls_{}'.format(cls), percent) for cls, percent in dev_label_distribution.items()))

        # create tokenizer and vocabulary

        sentences_train = variation_train_data['sentence']
        sentences_dev = variation_dev_data['sentence']

        word_tokenizer = Tokenizer(char_level=False)
        char_tokenizer = Tokenizer(char_level=True)
        word_tokenizer.fit_on_texts(sentences_train)
        char_tokenizer.fit_on_texts(sentences_train)
        print('Logging Info - Variation: {}, word_vocab: {}, char_vocab: {}'.format(variation,
                                                                                    len(word_tokenizer.word_index),
                                                                                    len(char_tokenizer.word_index)))
        analyze_result.update({'word_vocab': len(word_tokenizer.word_index),
                               'char_vocab': len(char_tokenizer.word_index)})

        # length analysis
        word_len_distribution, word_max_len = analyze_len_distribution(sentences_train, level='word')
        analyze_result.update(dict(('word_{}'.format(k), v) for k, v in word_len_distribution.items()))
        char_len_distribution, char_max_len = analyze_len_distribution(sentences_train, level='char')
        analyze_result.update(dict(('char_{}'.format(k), v) for k, v in char_len_distribution.items()))

        one_hot = False if config.loss_function == 'binary_crossentropy' else True
        train_word_ids = create_data_matrices(word_tokenizer, variation_train_data, config.n_class, one_hot,
                                              word_max_len)
        train_char_ids = create_data_matrices(char_tokenizer, variation_train_data, config.n_class, one_hot,
                                              char_max_len)
        dev_word_ids = create_data_matrices(word_tokenizer, variation_dev_data, config.n_class, one_hot, word_max_len)
        dev_char_ids = create_data_matrices(char_tokenizer, variation_dev_data, config.n_class, one_hot, char_max_len)

        # create embedding matrix by training on dataset
        w2v_data = train_w2v(sentences_train+sentences_dev, lambda x: x.split(), word_tokenizer.word_index)
        c2v_data = train_w2v(sentences_train+sentences_dev, lambda x: list(x), char_tokenizer.word_index)
        w_fasttext_data = train_fasttext(sentences_train+sentences_dev, lambda x: x.split(), word_tokenizer.word_index)
        c_fasttext_data = train_fasttext(sentences_train+sentences_dev, lambda x: list(x), char_tokenizer.word_index)
        w_glove_data = train_glove(sentences_train+sentences_dev, lambda x: x.split(), word_tokenizer.word_index)
        c_glove_data = train_glove(sentences_train+sentences_dev, lambda x: list(x), char_tokenizer.word_index)

        # save pre-process data
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, variation=variation), variation_train_data)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, variation=variation), variation_dev_data)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, variation=variation, level='word'),
                    train_word_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, variation=variation, level='char'),
                    train_char_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, variation=variation, level='word'),
                    dev_word_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, variation=variation, level='char'),
                    dev_char_ids)

        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation,
                                type='w2v_data'), w2v_data)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation,
                                type='c2v_data'), c2v_data)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation,
                                type='w_fasttext_data'), w_fasttext_data)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation,
                                type='c_fasttext_data'), c_fasttext_data)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation,
                                type='w_glove_data'), w_glove_data)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation,
                                type='c_glove_data'), c_glove_data)

        pickle_dump(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, variation=variation, level='word'),
                    word_tokenizer)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, variation=variation, level='char'),
                    char_tokenizer)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, variation=variation, level='word'),
                    word_tokenizer.word_index)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, variation=variation, level='char'),
                    char_tokenizer.word_index)

        # save analyze result
        write_log(format_filename(LOG_DIR, ANALYSIS_LOG_TEMPLATE, variation=variation), analyze_result)


if __name__ == '__main__':
    process_data()

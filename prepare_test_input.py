# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: prepare_test_input.py

@time: 2019/3/6 9:39

@desc:

"""

import os
import codecs
import itertools
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from config import PREDICT_DIR, SIMP_TEST_FILENAME, TRAD_TEST_FILENAME, TEST_DATA_TEMPLATE, TEST_IDS_MATRIX_TEMPLATE, \
    TEST_NGRAM_DATA_TEMPLATE, ModelConfig, TOKENIZER_TEMPLATE, VECTORIZER_TEMPLATE, PROCESSED_DATA_DIR
from utils.io import pickle_dump, pickle_load, format_filename


def read_raw_test_data(filename):
    raw_test_data = []
    with codecs.open(filename, 'r', encoding='utf8') as reader:
        for line in reader:
            raw_test_data.append(line.strip())

    return raw_test_data


def create_token_ids_matrix(tokenizer, sequences, max_len=None):
    tokens_ids = tokenizer.texts_to_sequences(sequences)

    # there might be zero len sequences - fix it by putting a random token there (or id 1 in the worst case)
    tokens_ids_flattened = list(itertools.chain.from_iterable(tokens_ids))
    max_id = max(tokens_ids_flattened) if len(tokens_ids_flattened) > 0 else -1

    for i in range(len(tokens_ids)):
        if len(tokens_ids[i]) == 0:
            print('Logging Warning - Unknown Sentence: {}'.format(sequences[i]))
            id_to_put = np.random.randint(1, max_id) if max_id != -1 else 1
            tokens_ids[i].append(id_to_put)

    print('Logging Info - pad sequence with max_len = %d', max_len)
    tokens_ids = pad_sequences(tokens_ids, maxlen=max_len)
    return tokens_ids


if __name__ == '__main__':
    if not os.path.exists(PREDICT_DIR):
        os.makedirs(PREDICT_DIR)
    config = ModelConfig()

    raw_data = dict()
    raw_data['simplified'] = read_raw_test_data(SIMP_TEST_FILENAME)
    raw_data['traditional'] = read_raw_test_data(TRAD_TEST_FILENAME)

    for variation in raw_data.keys():
        test_data = raw_data[variation]
        # prepare word embedding input
        word_tokenizer = pickle_load(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, variation=variation,
                                                     level='word'))
        word_ids_test = create_token_ids_matrix(word_tokenizer, raw_data[variation], config.word_max_len)

        # prepare n-gram input
        vectorizer = pickle_load(format_filename(PROCESSED_DATA_DIR, VECTORIZER_TEMPLATE, variation=variation,
                                                 type='binary', level='char', ngram_range=(2, 3)))
        n_gram_test = vectorizer.transform(raw_data[variation])

        pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, variation=variation), {'sentence': test_data})
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_IDS_MATRIX_TEMPLATE, variation=variation, level='word'),
                    {'sentence': word_ids_test})
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_NGRAM_DATA_TEMPLATE, variation=variation, type='binary',
                                    level='char', ngram_range=(2, 3)), {'sentence': n_gram_test})

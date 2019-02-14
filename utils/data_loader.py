# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_loader.py

@time: 2019/2/9 13:38

@desc:

"""

import codecs
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix, csr_matrix, csc_matrix

from config import PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, TEST_IDS_MATRIX_TEMPLATE, \
    TRAIN_NGRAM_DATA_TEMPLATE, DEV_NGRAM_DATA_TEMPLATE, TEST_NGRAM_DATA_TEMPLATE, LABELS
from utils.io import pickle_load, format_filename


def read_raw_data(filename, set_variation=None):
    all_rows = []
    with codecs.open(filename, 'r', encoding='utf8') as reader:
        for i, line in enumerate(reader):
            row = dict()
            line_split = line.strip().split()
            sentence = line_split[:-1]
            label = line_split[-1]
            if label not in LABELS:
                print('Logging Warning - Line %d not contain correct label: %s', i+1, line)
                continue
            row['sentence'] = ' '.join(sentence)
            row['label'] = label
            all_rows.append(row)
    raw_data = pd.DataFrame(all_rows)

    if set_variation is not None:
        raw_data['variation'] = set_variation

    return raw_data


def load_processed_data(variation, level, data_type):
    if data_type == 'train':
        filename = format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, variation=variation, level=level)
    elif data_type == 'valid' or data_type == 'dev':
        filename = format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, variation=variation, level=level)
    elif data_type == 'test':
        filename = format_filename(PROCESSED_DATA_DIR, TEST_IDS_MATRIX_TEMPLATE, variation=variation, level=level)
    else:
        raise ValueError('Data Type Not Understood: {}'.format(data_type))
    return pickle_load(filename)


def load_single_ngram_data(variation, vectorizer_type, level, ngram_range, data_type):
    if data_type == 'train':
        filename = format_filename(PROCESSED_DATA_DIR, TRAIN_NGRAM_DATA_TEMPLATE, variation=variation,
                                   type=vectorizer_type, level=level, ngram_range=ngram_range)
    elif data_type == 'valid' or data_type == 'dev':
        filename = format_filename(PROCESSED_DATA_DIR, DEV_NGRAM_DATA_TEMPLATE, variation=variation,
                                   type=vectorizer_type, level=level, ngram_range=ngram_range)
    elif data_type == 'test':
        filename = format_filename(PROCESSED_DATA_DIR, TEST_NGRAM_DATA_TEMPLATE, variation=variation,
                                   type=vectorizer_type, level=level, ngram_range=ngram_range)
    else:
        raise ValueError('Data Type Not Understood: {}'.format(data_type))
    return pickle_load(filename)


def load_ngram_data(variation, vectorizer_type, level, ngram_range, data_type):
    if isinstance(level, list):     # concatenate multiple input
        if not isinstance(ngram_range, list):
            ngram_range = [ngram_range]*len(level)
        elif len(ngram_range) != len(level):
            raise ValueError('size of `level` list and `ngram_range` list should be equal')
        all_data = [load_single_ngram_data(variation, vectorizer_type, level[i], ngram_range[i], data_type)
                    for i in range(len(level))]
        ngram_data = {}
        if isinstance(all_data[0]['sentence'], (coo_matrix, csc_matrix, csr_matrix)):
            ngram_data['sentence'] = hstack([data_chunk['sentence'] for data_chunk in all_data])
        else:
            ngram_data['sentence'] = np.concatenate([data_chunk['sentence'] for data_chunk in all_data])
        ngram_data['label'] = all_data[0]['label']
        return ngram_data
    else:
        return load_single_ngram_data(variation, vectorizer_type, level, ngram_range, data_type)




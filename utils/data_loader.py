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
import pandas as pd

from config import PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, TEST_IDS_MATRIX_TEMPLATE, \
    LABELS
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

# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_augment.py

@time: 2019/2/14 21:56

@desc:

"""

import numpy as np

from config import PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, VARIATIONS, TOKENIZER_TEMPLATE, \
    TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, LOG_DIR, ANALYSIS_LOG_TEMPLATE, EMBEDDING_MATRIX_TEMPLATE
from config import ModelConfig
from utils.io import pickle_load, format_filename, pickle_dump, write_log
from preprocess import create_data_matrices, analyze_len_distribution
from prepare_ngram_feature import prepare_ngram_feature


def augment_data(data, ignore_short_messages=3, double_long_messages=10, triple_very_long_messages=15):
    sentences_augment, labels_augment = [], []
    for sentence, label in zip(data['sentence'], data['label']):
        words = sentence.split(' ')
        length = len(words)
        if length < ignore_short_messages:
            continue
        if length >= triple_very_long_messages:
            offset = round(length / 3)
            half_offset = round(length / 6)
            s1 = ' '.join(words[:-offset])
            s2 = ' '.join(words[offset:])
            s3 = ' '.join(words[half_offset:-half_offset])
            # add three new sentences instead of the one old one
            sentences_augment.extend([s1, s2, s3])
            labels_augment.extend([label, label, label])
        elif length >= double_long_messages:
            offset = round(length / 4)
            s1 = ' '.join(words[:-offset])
            s2 = ' '.join(words[offset:])
            # add two new sentences instead of the one old one
            sentences_augment.extend([s1, s2])
            labels_augment.extend([label, label])
        else:
            sentences_augment.append(sentence)
            labels_augment.append(label)
    data_augment = {'sentence': sentences_augment, 'label': labels_augment}
    return data_augment


def process_data():
    config = ModelConfig()
    for variation in VARIATIONS:
        train_data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, variation=variation))
        dev_data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, variation=variation))
        word_tokenizer = pickle_load(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, variation=variation,
                                                     level='word'))
        char_tokenizer = pickle_load(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, variation=variation,
                                                     level='char'))
        w2v_data = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation, type='w2v_data'))
        c2v_data = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation, type='c2v_data'))

        train_data_augment = augment_data(train_data)

        analyze_result = {}
        analyze_result.update({'train_set': len(train_data['sentence']),
                               'train_augment_set': len(train_data_augment['sentence']),
                               'dev_set': len(dev_data['sentence'])})
        word_len_distribution, word_max_len = analyze_len_distribution(train_data_augment['sentence'], level='word')
        analyze_result.update(dict(('word_{}'.format(k), v) for k, v in word_len_distribution.items()))
        char_len_distribution, char_max_len = analyze_len_distribution(train_data_augment['sentence'], level='char')
        analyze_result.update(dict(('char_{}'.format(k), v) for k, v in char_len_distribution.items()))

        one_hot = False if config.loss_function == 'binary_crossentropy' else True
        train_word_ids = create_data_matrices(word_tokenizer, train_data_augment, config.n_class, one_hot,
                                              word_max_len)
        train_char_ids = create_data_matrices(char_tokenizer, train_data_augment, config.n_class, one_hot,
                                              char_max_len)
        dev_word_ids = create_data_matrices(word_tokenizer, dev_data, config.n_class, one_hot, word_max_len)
        dev_char_ids = create_data_matrices(char_tokenizer, dev_data, config.n_class, one_hot, char_max_len)

        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, variation=variation+'_aug'),
                    train_data_augment)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, variation=variation+'_aug'), dev_data)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, variation=variation+'_aug',
                                    level='word'), train_word_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, variation=variation+'_aug',
                                    level='char'), train_char_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, variation=variation+'_aug',
                                    level='word'), dev_word_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, variation=variation+'_aug',
                                    level='char'), dev_char_ids)

        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation+'_aug',
                                type='w2v_data'), w2v_data)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, variation=variation+'_aug',
                                type='c2v_data'), c2v_data)

        prepare_ngram_feature('binary', 'char', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char', (2, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char_wb', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char_wb', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char_wb', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char_wb', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char_wb', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'char_wb', (2, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'word', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'word', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'word', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'word', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'word', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('binary', 'word', (2, 3), train_data_augment, dev_data, variation+'_aug')

        prepare_ngram_feature('tf', 'char', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char', (2, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char_wb', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char_wb', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char_wb', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char_wb', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char_wb', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'char_wb', (2, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'word', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'word', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'word', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'word', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'word', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tf', 'word', (2, 3), train_data_augment, dev_data, variation+'_aug')

        prepare_ngram_feature('tfidf', 'char', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char', (2, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char_wb', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char_wb', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char_wb', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char_wb', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char_wb', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'char_wb', (2, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'word', (1, 1), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'word', (2, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'word', (3, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'word', (1, 2), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'word', (1, 3), train_data_augment, dev_data, variation+'_aug')
        prepare_ngram_feature('tfidf', 'word', (2, 3), train_data_augment, dev_data, variation+'_aug')

        # save analyze result
        write_log(format_filename(LOG_DIR, ANALYSIS_LOG_TEMPLATE, variation=variation+'_aug'), analyze_result)


if __name__ == '__main__':
    process_data()


# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: prepare_ngram_feature.py

@time: 2019/2/13 16:26

@desc:

"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import VARIATIONS, PROCESSED_DATA_DIR, LOG_DIR, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, \
    TRAIN_NGRAM_DATA_TEMPLATE, DEV_NGRAM_DATA_TEMPLATE, VECTORIZER_TEMPLATE

from utils.io import pickle_load, pickle_dump, format_filename


def prepare_ngram_feature(vectorizer_type, level, ngram_range, train_data, dev_data, variation):
    if level not in ['word', 'char', 'char_wb']:
        raise ValueError('Vectorizer Level Not Understood: {}'.format(level))
    if not isinstance(ngram_range, tuple):
        raise ValueError('ngram_range should be a tuple, got {}'.format(type(ngram_range)))
    if vectorizer_type == 'binary':
        vectorizer = CountVectorizer(binary=True, analyzer=level, ngram_range=ngram_range)
    elif vectorizer_type == 'tf':
        vectorizer = CountVectorizer(binary=False, analyzer=level, ngram_range=ngram_range)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer=level, ngram_range=ngram_range)
    else:
        raise ValueError('Vectorizer Type Not Understood: {}'.format(vectorizer_type))

    train_ngram_feature = vectorizer.fit_transform(train_data['sentence'])
    train_ngram_data = {'sentence': train_ngram_feature, 'label': train_data['label']}

    dev_ngram_feature = vectorizer.transform(dev_data['sentence'])
    dev_ngram_data = {'sentence': dev_ngram_feature, 'label': dev_data['label']}

    print('Logging info - {}_{}vectorizer_{}_{} : train_ngram_feature shape: {}, '
          'dev_ngram_feature shape: {}'.format(variation, vectorizer_type, level, ngram_range,
                                               train_ngram_feature.shape, dev_ngram_feature.shape))

    pickle_dump(format_filename(PROCESSED_DATA_DIR, VECTORIZER_TEMPLATE, variation=variation, type=vectorizer_type,
                                level=level, ngram_range=ngram_range), vectorizer)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_NGRAM_DATA_TEMPLATE, variation=variation,
                                type=vectorizer_type, level=level, ngram_range=ngram_range), train_ngram_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_NGRAM_DATA_TEMPLATE, variation=variation,
                                type=vectorizer_type, level=level, ngram_range=ngram_range), dev_ngram_data)
    return vectorizer, train_ngram_data, dev_ngram_data


def process_data():
    for variation in VARIATIONS:
        train_data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, variation=variation))
        dev_data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, variation=variation))

        # prepare_ngram_feature('binary', 'char', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char', (2, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char_wb', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char_wb', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char_wb', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char_wb', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char_wb', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'char_wb', (2, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'word', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'word', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'word', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'word', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'word', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('binary', 'word', (2, 3), train_data, dev_data, variation)
        #
        # prepare_ngram_feature('tf', 'char', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char', (2, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char_wb', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char_wb', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char_wb', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char_wb', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char_wb', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'char_wb', (2, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'word', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'word', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'word', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'word', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'word', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tf', 'word', (2, 3), train_data, dev_data, variation)
        #
        # prepare_ngram_feature('tfidf', 'char', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char', (2, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char_wb', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char_wb', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char_wb', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char_wb', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char_wb', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'char_wb', (2, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'word', (1, 1), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'word', (2, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'word', (3, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'word', (1, 2), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'word', (1, 3), train_data, dev_data, variation)
        # prepare_ngram_feature('tfidf', 'word', (2, 3), train_data, dev_data, variation)

        # prepare_ngram_feature('binary', 'char', (4, 4), train_data, dev_data, variation)


if __name__ == '__main__':
    process_data()


# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: prepare_ngram_feature.py

@time: 2019/2/13 16:26

@desc:

"""

import dill
from jieba import posseg
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from config import VARIATIONS, PROCESSED_DATA_DIR, LOG_DIR, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, \
    TRAIN_NGRAM_DATA_TEMPLATE, DEV_NGRAM_DATA_TEMPLATE, VECTORIZER_TEMPLATE

from utils.io import pickle_load, pickle_dump, format_filename


def basic_tokenize(sentence, analyzer='word'):
    if analyzer == 'word':
        return sentence.split()
    else:
        sentence = sentence.replace(' ', '')
        return list(sentence)


def skipgram_tokenize(sentence, n=None, k=None, include_all=False, analyzer='word'):
    from nltk.util import skipgrams
    tokens = [w for w in basic_tokenize(sentence, analyzer)]
    if include_all:
        result = []
        for i in range(k+1):
            skg = [w for w in skipgrams(tokens, n, i)]
            result = result+skg
    else:
        result = [w for w in skipgrams(tokens, n, k)]
    result = set(result)
    return result


def make_skip_tokenize(n, k, analyzer='word', include_all=False):
    return lambda x: skipgram_tokenize(x, n=n, k=k, analyzer=analyzer, include_all=include_all)


def prepare_skip_ngram_feature(vectorizer_type, level, ngram, skip_k, train_data, dev_data, variation):
    if level not in ['word', 'char']:
        raise ValueError('Vectorizer Level Not Understood: {}'.format(level))

    if vectorizer_type == 'binary':
        vectorizer = CountVectorizer(binary=True, tokenizer=make_skip_tokenize(ngram, skip_k, level))
    elif vectorizer_type == 'tf':
        vectorizer = CountVectorizer(binary=False, tokenizer=make_skip_tokenize(ngram, skip_k, level))
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(make_skip_tokenize(ngram, skip_k, level))
    else:
        raise ValueError('Vectorizer Type Not Understood: {}'.format(vectorizer_type))

    train_ngram_feature = vectorizer.fit_transform(train_data['sentence'])
    train_ngram_data = {'sentence': train_ngram_feature, 'label': train_data['label']}

    dev_ngram_feature = vectorizer.transform(dev_data['sentence'])
    dev_ngram_data = {'sentence': dev_ngram_feature, 'label': dev_data['label']}

    print('Logging info - {}_{}vectorizer_{}_{}_{} : train_skip_ngram_feature shape: {}, '
          'dev_skip_ngram_feature shape: {}'.format(variation, vectorizer_type, level, ngram, skip_k,
                                                    train_ngram_feature.shape, dev_ngram_feature.shape))

    # pickle can't pickle lambda function, here i use drill: https://github.com/uqfoundation/dill
    with open(format_filename(PROCESSED_DATA_DIR, VECTORIZER_TEMPLATE, variation=variation, type=vectorizer_type,
                              level=level, ngram_range='%d_%d' % (ngram, skip_k)), 'wb') as writer:

        dill.dump(vectorizer, writer)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_NGRAM_DATA_TEMPLATE, variation=variation,
                                type=vectorizer_type, level=level, ngram_range='%d_%d' % (ngram, skip_k)),
                train_ngram_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_NGRAM_DATA_TEMPLATE, variation=variation,
                                type=vectorizer_type, level=level, ngram_range='%d_%d' % (ngram, skip_k)),
                dev_ngram_data)
    return vectorizer, train_ngram_data, dev_ngram_data


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


def get_pos(sentence):
    return ' '.join([pos for word, pos in posseg.cut(sentence) if word != ' '])


def process_data():
    for variation in VARIATIONS:
        train_data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, variation=variation))
        dev_data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, variation=variation))

        # prepare ngram feature
        for vectorizer_type in ['binary']:
            for level in ['char', 'word']:
                for ngram_range in [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
                    prepare_ngram_feature(vectorizer_type, level, ngram_range, train_data, dev_data, variation)
        #
        # # prepare skip ngram features
        # for vectorizer_type in ['binary', 'tf', 'tfidf']:
        #     for level in ['word', 'char']:
        #         for ngram in [2, 3]:
        #             for skip_k in [1, 2, 3]:
        #                 prepare_skip_ngram_feature(vectorizer_type, level, ngram, skip_k, train_data, dev_data,
        #                                            variation)

        # prepare pos ngram
        # train_pos_data = {'sentence': [get_pos(sentence) for sentence in train_data['sentence']],
        #                   'label': train_data['label']}
        # dev_pos_data = {'sentence': [get_pos(sentence) for sentence in dev_data['sentence']],
        #                 'label': dev_data['label']}
        # for vectorizer_type in ['binary', 'tf', 'tfidf']:
        #     for level in ['word']:
        #         for ngram_range in [(1, 1), (2, 2), (3, 3)]:
        #             prepare_ngram_feature(vectorizer_type, level, ngram_range, train_pos_data, dev_pos_data,
        #                                   variation+'_pos')


if __name__ == '__main__':
    process_data()


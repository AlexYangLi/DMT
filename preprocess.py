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
import dill
from jieba import posseg
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from config import SIMP_TRAIN_FILENAME, SIMP_DEV_FILENAME, TRAD_TRAIN_FILENAME, TRAD_DEV_FILENAME, TRAIN_DATA_TEMPLATE,\
    DEV_DATA_TEMPLATE, TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, EMBEDDING_MATRIX_TEMPLATE, \
    TOKENIZER_TEMPLATE, VOCABULARY_TEMPLATE, ANALYSIS_LOG_TEMPLATE, TRAIN_NGRAM_DATA_TEMPLATE, DEV_NGRAM_DATA_TEMPLATE,\
    VECTORIZER_TEMPLATE
from config import PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, IMG_DIR
from config import LABELS, VARIATIONS
from config import ModelConfig
from utils.data_loader import read_raw_data
from utils.analysis import analyze_len_distribution, analyze_class_distribution
from utils.embeddings import train_w2v, train_glove, train_fasttext
from utils.io import pickle_dump, format_filename, write_log


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


def get_sentence_label(data):
    labels = data['label'].map(LABELS).tolist()
    sentences = data['sentence'].tolist()
    return {'sentence': sentences, 'label': labels}


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


def skipgram_tokenize(sentence, n=None, k=None, include_all=False, analyzer='word'):
    def basic_tokenize(sentence, analyzer='word'):
        if analyzer == 'word':
            return sentence.split()
        else:
            sentence = sentence.replace(' ', '')
            return list(sentence)

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

        if config.data_augment:
            variation_train_data = augment_data(variation_train_data)
            variation += '_aug'

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

        # prepare ngram feature
        for vectorizer_type in ['binary', 'tf', 'tfidf']:
            for level in ['char', 'word']:
                for ngram_range in [(1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (2, 4), (1, 4), (4, 4), (5, 5), (6, 6),
                                    (7, 7), (8, 8)]:
                    prepare_ngram_feature(vectorizer_type, level, ngram_range, variation_train_data, variation_dev_data,
                                          variation)

        # prepare skip ngram features
        for vectorizer_type in ['binary', 'tf', 'tfidf']:
            for level in ['word', 'char']:
                for ngram in [2, 3]:
                    for skip_k in [1, 2, 3]:
                        prepare_skip_ngram_feature(vectorizer_type, level, ngram, skip_k, variation_train_data,
                                                   variation_dev_data, variation)

        # prepare pos ngram
        variation_train_pos_data = {'sentence': [get_pos(sentence) for sentence in variation_train_data['sentence']],
                                    'label': variation_train_data['label']}
        variation_dev_pos_data = {'sentence': [get_pos(sentence) for sentence in variation_dev_data['sentence']],
                                  'label': variation_dev_data['label']}
        for vectorizer_type in ['binary', 'tf', 'tfidf']:
            for level in ['word']:
                for ngram_range in [(1, 1), (2, 2), (3, 3)]:
                    prepare_ngram_feature(vectorizer_type, level, ngram_range, variation_train_pos_data,
                                          variation_dev_pos_data, variation+'_pos')

        # save analyze result
        write_log(format_filename(LOG_DIR, ANALYSIS_LOG_TEMPLATE, variation=variation), analyze_result)


if __name__ == '__main__':
    process_data()

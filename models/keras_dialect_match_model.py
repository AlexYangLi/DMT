# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_dialect_match_model.py

@time: 2019/2/15 15:36

@desc:

"""

import numpy as np

from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, Lambda, GRU, GlobalAveragePooling1D, \
    concatenate, Conv1D, GlobalMaxPooling1D, Dense, Flatten, multiply
from keras import Model, backend as K
from keras.callbacks import Callback

from models.keras_base_model import KerasBaseModel
from layers.attention import SelfAttention, AdditiveAttention, DotProductAttention, MultiplicativeAttention, \
    ConcatAttention
from utils.metrics import eval_acc, eval_f1


class DialectMatchMetrics(Callback):
    def __init__(self, dev_data):
        self.dev_sentences = dev_data['sentence']
        self.dev_label = dev_data['label']
        self.dev_size = self.dev_sentences.shape[0]
        super(DialectMatchMetrics, self).__init__()

    def on_train_begin(self, logs=None):
        self.val_accs = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        predict_t = self.model.predict([self.dev_sentences, np.zeros(shape=(self.dev_size, 1))])
        predict_m = self.model.predict([self.dev_sentences, np.ones(shape=(self.dev_size, 1))])
        dev_predict = np.argmax(np.concatenate((predict_t, predict_m), axis=-1), axis=-1)
        _val_acc = eval_acc(self.dev_label, dev_predict)
        _val_f1 = eval_f1(self.dev_label, dev_predict)
        logs['val_match_acc'] = _val_acc
        logs['val_match_f1'] = _val_f1
        self.val_accs.append(_val_acc)
        self.val_f1s.append(_val_f1)
        print('val_match_acc: %f' % _val_acc)
        print('val_match_f1: %f' % _val_f1)


class DialectMatchModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(DialectMatchModel, self).__init__(config, **kwargs)

    def build(self, encoder_type='concat_attention', metrics='euclidean'):
        input_text = Input(shape=(self.max_len,))
        input_dialect = Input(shape=(1,))

        word_embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                         weights=[self.word_embeddings],
                                         trainable=self.config.word_embed_trainable, mask_zero=True)
        dialect_embedding_layer = Embedding(2, self.config.word_embeddings.shape[1], trainable=True)

        text_embed = SpatialDropout1D(0.2)(word_embedding_layer(input_text))
        dialect_embed = Flatten()(dialect_embedding_layer(input_dialect))

        text_encode, dialect_encode = self.sentence_encoder(text_embed, dialect_embed, encoder_type)

        if metrics == 'sigmoid':
            t_sub_d = Lambda(lambda x: K.abs(x[0] - x[1]))([text_encode, dialect_encode])
            t_mul_d = multiply([text_encode, dialect_encode])
            concat_encode = concatenate([text_encode, dialect_encode, t_sub_d, t_mul_d])

            dense = Dense(units=self.config.dense_units, activation='relu')(concat_encode)
            distance = Dense(1, activation='sigmoid')(dense)
            loss = 'binary_crossentropy'
        elif metrics == 'manhattan':
            distance = Lambda(self.manhattan_distance)([text_encode, dialect_encode])
            loss = 'binary_crossentropy'
        elif metrics == 'euclidean':
            distance = Lambda(self.euclidean_distance)([text_encode, dialect_encode])
            loss = self.contrastive_loss
        else:
            raise ValueError('Similarity Metrics Not Understood: {}'.format(metrics))

        model = Model([input_text, input_dialect], distance)
        model.compile(loss=loss, metrics=['acc'], optimizer=self.config.optimizer)
        return model

    def sentence_encoder(self, text_embed, dialect_embed, encoder_type):
        if encoder_type == 'lstm':
            text_encode = LSTM(units=self.config.rnn_units)(text_embed)
            dialect_encode = Dense(units=self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'gru':
            text_encode = GRU(units=self.config.rnn_units)(text_embed)
            dialect_encode = Dense(units=self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'bilstm':
            text_encode = Bidirectional(LSTM(units=self.config.rnn_units))(text_embed)
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'bigru':
            text_encode = Bidirectional(GRU(units=self.config.rnn_units))(text_embed)
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'bilstm_max_pool':
            bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
            global_max_pooling = Lambda(lambda x: K.max(x, axis=1))     # GlobalMaxPooling1D didn't support masking
            text_encode = global_max_pooling(bilstm(text_embed))
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'bilstm_mean_pool':
            bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
            text_encode = GlobalAveragePooling1D()(bilstm(text_embed))
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'self_attention':
            bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
            text_encode = SelfAttention()(bilstm(text_embed))
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'h_cnn':
            cnn_text = [text_embed]
            filter_lengths = [2, 3, 4, 5]
            for filter_length in filter_lengths:
                conv_layer = Conv1D(filters=self.config.rnn_units, kernel_size=filter_length, padding='valid',
                                    strides=1, activation='relu')
                cnn_text.append(conv_layer(cnn_text[-1]))
            global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
            cnn_text = [global_max_pooling(cnn_text[i]) for i in range(1, 5)]
            text_encode = concatenate(cnn_text)
            dialect_encode = Dense(units=4 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'cnn':
            filter_lengths = [2, 3, 4, 5]
            conv_layers = []
            for filter_length in filter_lengths:
                conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                    strides=1, activation='relu')(text_embed)
                maxpooling = GlobalMaxPooling1D()(conv_layer)
                conv_layers.append(maxpooling)
            text_encode = concatenate(conv_layers)
            dialect_encode = Dense(units=4 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'dot_attention':
            text_hidden = LSTM(units=self.config.rnn_units, return_sequences=True)(text_embed)
            text_encode = DotProductAttention()([text_hidden, dialect_embed])
            dialect_encode = Dense(units=self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'mul_attention':
            text_hidden = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))(text_embed)
            text_encode = MultiplicativeAttention()([text_hidden, dialect_embed])
            dialect_encode = Dense(units=2*self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'add_attention':
            text_hidden = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))(text_embed)
            text_encode = AdditiveAttention()([text_hidden, dialect_embed])
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        elif encoder_type == 'concat_attention':
            text_hidden = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))(text_embed)
            text_encode = ConcatAttention()([text_hidden, dialect_embed])
            dialect_encode = Dense(units=2 * self.config.rnn_units, activation='relu')(dialect_embed)
        else:
            raise ValueError('Encoder Type Not Understood : {}'.format(encoder_type))

        return text_encode, dialect_encode

    def train(self, data_train, data_dev=None):
        match_data_train = self.prepare_match_input(data_train)
        match_data_dev = self.prepare_match_input(data_dev)

        self.callbacks.append(DialectMatchMetrics(data_dev))
        print('Logging Info - start training...')
        x_train, y_train = match_data_train['sentence'], match_data_train['label']
        if data_dev is None:
            self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                           validation_split=0.1, callbacks=self.callbacks)
        else:
            x_dev, y_dev = match_data_dev['sentence'], match_data_dev['label']
            self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                           validation_data=(x_dev, y_dev), callbacks=self.callbacks)
        print('Logging Info - training end...')

    def evaluate(self, data):
        predictions = self.predict(data['sentence'])
        predictions = np.argmax(predictions, axis=-1)
        labels = data['label']

        acc = eval_acc(labels, predictions)
        f1 = eval_f1(labels, predictions)
        print('acc : {}, f1 : {}'.format(acc, f1))
        return acc, f1

    def predict(self, data):
        data_size = data.shape[0]
        predict_t = self.model.predict([data, np.zeros(shape=(data_size, 1))])
        predict_m = self.model.predict([data, np.ones(shape=(data_size, 1))])
        dev_predict = np.concatenate((predict_t, predict_m), axis=-1)
        return dev_predict

    @staticmethod
    def manhattan_distance(vectors):
        x, y = vectors
        return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

    @staticmethod
    def euclidean_distance(vectors):
        x, y = vectors
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    @staticmethod
    def prepare_match_input(input_data):
        sentences, labels = input_data['sentence'], input_data['label']
        match_sentences, match_dialects, match_labels = [], [], []

        for sentence, label in zip(sentences, labels):
            if label == 1:  # mainland
                match_sentences.append(sentence)
                match_dialects.append([0])  # dialect indeex 0 for taiwan
                match_labels.append([0])

                match_sentences.append(sentence)
                match_dialects.append([0])  # dialect indeex 1 for mainland
                match_labels.append([1])
            elif label == 0:
                match_sentences.append(sentence)
                match_dialects.append([0])  # dialect indeex 0 for taiwan
                match_labels.append([1])

                match_sentences.append(sentence)
                match_dialects.append([0])  # dialect indeex 1 for mainland
                match_labels.append([0])
            else:
                raise ValueError('Invalid Input: {}'.format(label))
        match_data = {'sentence': [np.array(match_sentences), np.array(match_dialects)],
                      'label': np.array(match_labels)}
        return match_data


# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_base_model.py

@time: 2019/2/3 17:14

@desc:

"""

import os
import abc
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.base_model import BaseModel
from utils.metrics import eval_acc, eval_f1, return_error_index


class KerasBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(KerasBaseModel, self).__init__()
        self.config = config
        self.level = self.config.input_level
        self.max_len = self.config.max_len[self.config.input_level]
        self.word_embeddings = config.word_embeddings
        self.n_class = config.n_class

        self.callbacks = []
        self.init_callbacks()

        self.model = self.build(**kwargs)

    def init_callbacks(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))

        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))

    def load_model(self, filename):
        self.model.load_weights(filename)

    def load_best_model(self):
        print('Logging Info - loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        print('Logging Info - Model loaded')

    @abc.abstractmethod
    def build(self, **kwargs):
        """Build the model"""

    def train(self, data_train, data_dev=None):
        print('Logging Info - start training...')
        x_train, y_train = data_train['sentence'], data_train['label']
        if data_dev is None:
            self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                           validation_split=0.1, callbacks=self.callbacks)
        else:
            x_dev, y_dev = data_dev['sentence'], data_dev['label']
            self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                           validation_data=(x_dev, y_dev), callbacks=self.callbacks)
        print('Logging Info - training end...')

    def evaluate(self, data):
        predictions = self.predict(data)
        if self.config.loss_function == 'binary_crossentropy':
            labels = data['label']
        else:
            labels = np.argmax(data['label'], axis=-1)

        acc = eval_acc(labels, predictions)
        f1 = eval_f1(labels, predictions)
        print('acc : {}, f1 : {}'.format(acc, f1))
        return acc, f1

    def predict(self, data):
        predictions = self.predict_proba(data)
        if self.config.loss_function == 'binary_crossentropy':
            return np.array([1 if prediction >= self.config.binary_threshold else 0 for prediction in predictions])
        else:
            return np.argmax(predictions, axis=-1)

    def predict_proba(self, data):
        pred_proba = self.model.predict(data['sentence'])
        if self.config.loss_function == 'binary_crossentropy':
            pred_proba = pred_proba.flatten()
        return pred_proba

    def error_analyze(self, data):
        pred_probas = self.predict_proba(data)
        if self.config.loss_function == 'binary_crossentropy':
            pred_labels = np.array([1 if pred_prob >= self.config.binary_threshold else 0 for pred_prob in pred_probas])
        else:
            pred_labels = np.argmax(pred_probas, axis=-1)
        if self.config.loss_function == 'binary_crossentropy':
            labels = data['label']
        else:
            labels = np.argmax(data['label'], axis=-1)
        error_index = return_error_index(labels, pred_labels)

        return error_index, pred_probas[error_index]

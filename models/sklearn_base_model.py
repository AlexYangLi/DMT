# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: sklearn_base_model.py

@time: 2019/2/13 14:29

@desc:

"""
import os
import abc

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from models.base_model import BaseModel
from utils.metrics import eval_acc, eval_f1, eval_precision, eval_recall


class SklearnBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(SklearnBaseModel, self).__init__()
        self.config = config
        self.model = self.build(**kwargs)

    @abc.abstractmethod
    def build(self, **kwargs):
        """Build the model"""

    def load_model(self, filename):
        self.model = joblib.load(filename)

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_best_model(self):
        print('loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        print('Model loaded')

    def train(self, data_train, data_dev=None):
        print('Logging Info - start training...')
        x_train, y_train = data_train['sentence'], data_train['label']
        self.model.fit(x_train, y_train)
        print('Logging Info - training end...')

        print('saving model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.save_model(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        print('Model saved')

    def evaluate(self, data):
        predictions = self.predict(data['sentence'])
        labels = data['label']

        acc = eval_acc(labels, predictions)
        f1 = eval_f1(labels, predictions)
        p = eval_precision(labels, predictions)
        r = eval_recall(labels, predictions)
        print('acc: {}, f1: {}, p: {}, r: {}'.format(acc, f1, p, r))
        return acc, f1, p, r

    def predict(self, data):
        return self.model.predict(data)


class SVMModel(SklearnBaseModel):
    def build(self, **kwargs):
        return LinearSVC(**kwargs)


class LRModel(SklearnBaseModel):
    def build(self, **kwargs):
        return LogisticRegression(**kwargs)


class SGDModel(SklearnBaseModel):
    def build(self, **kwargs):
        return SGDClassifier(**kwargs)


class GaussianNBModel(SklearnBaseModel):
    def build(self, **kwargs):
        return GaussianNB(**kwargs)


class MultinomialNBModel(SklearnBaseModel):
    def build(self, **kwargs):
        return MultinomialNB(**kwargs)


class BernoulliNBModel(SklearnBaseModel):
    def build(self, **kwargs):
        return BernoulliNB(**kwargs)


class RandomForestModel(SklearnBaseModel):
    def build(self, **kwargs):
        return RandomForestClassifier(**kwargs)


class GBDTModel(SklearnBaseModel):
    def build(self, **kwargs):
        return GradientBoostingClassifier(**kwargs)


class XGBoostModel(SklearnBaseModel):
    def build(self, **kwargs):
        return XGBClassifier(**kwargs)



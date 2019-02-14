# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_han_model.py

@time: 2019/2/8 13:22

@desc:

"""


from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, GRU, Masking, TimeDistributed

from models.keras_base_model import KerasBaseModel
from layers.attention import SelfAttention


class HAN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(HAN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len, ))

        sent_encoded = self.word_encoder()(input_text)  # word encoder
        sent_vector = SelfAttention(bias=True)(sent_encoded)  # word attention

        dense_layer = Dense(256, activation='relu')(sent_vector)
        if self.config.loss_function == 'binary_crossentropy':
            output = Dense(1, activation='sigmoid')(dense_layer)
        else:
            output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss=self.config.loss_function, metrics=['acc'], optimizer=self.config.optimizer)
        return model

    def word_encoder(self):
        input_words = Input(shape=(self.max_len,))
        word_vectors = Embedding(input_dim=self.word_embeddings.shape[0], output_dim=self.word_embeddings.shape[1],
                                 weights=[self.word_embeddings], mask_zero=True,
                                 trainable=self.config.word_embed_trainable)(input_words)
        sent_encoded = Bidirectional(GRU(self.config.rnn_units, return_sequences=True))(word_vectors)
        return Model(input_words, sent_encoded)

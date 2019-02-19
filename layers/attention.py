# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: attention.py

@time: 2019/2/8 13:36

@desc: attention mechanism, support masking

"""

from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer


class SelfAttention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
 e: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, bias=False, return_score=False, **kwargs):
        self.supports_masking = True
        self.return_score = return_score

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = SelfAttention.dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_score:
            return K.sum(weighted_input, axis=1), a
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)


class AdditiveAttention(Layer):
    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(AdditiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, query_shape = input_shape
        if len(context_shape) != 3:
            raise ValueError('Context input into AdditiveAttention should be a 3D tensor')
        if len(query_shape) != 2:
            raise ValueError('Query input into AdditiveAttention should be a 2D tensor')

        self.context_w = self.add_weight(shape=(context_shape[-1], context_shape[-1]), initializer=self.initializer,
                                         regularizer=self.regularizer, constraint=self.constraint,
                                         name='{}_context_w'.format(self.name))
        self.query_w = self.add_weight(shape=(query_shape[-1], context_shape[-1]), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint,
                                       name='{}_query_w'.format(self.name))
        self.attend_w = self.add_weight(shape=(context_shape[-1], 1), initializer=self.initializer,
                                        regularizer=self.regularizer, constraint=self.constraint,
                                        name='{}_attend_w'.format(self.name))
        super(AdditiveAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        context, query = inputs
        if mask is None:
            context_mask = None
        else:
            context_mask, _ = mask

        time_step = K.shape(context)[1]

        repeat_query = K.repeat(query, time_step)

        g = K.dot(K.tanh(K.dot(context, self.context_w) + K.dot(repeat_query, self.query_w)), self.attend_w)
        a = K.exp(K.squeeze(g, axis=-1))

        if context_mask is not None:
            a *= K.cast(context_mask, K.floatx())

        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())     # [batch_size, time_steps]

        # apply attention
        a_expand = K.expand_dims(a)  # [batch_size, time_steps, 1]
        attend_context = K.sum(context * a_expand, axis=1)  # [batch_size, hidden]

        if self.return_attend_weight:
            return attend_context, a
        else:
            return attend_context

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, _ = input_shape

        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]), (context_shape[0], context_shape[1])]
        else:
            return context_shape[0], context_shape[-1]


class MultiplicativeAttention(Layer):
    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(MultiplicativeAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, query_shape = input_shape
        if len(context_shape) != 3:
            raise ValueError('Context input into AdditiveAttention should be a 3D tensor')
        if len(query_shape) != 2:
            raise ValueError('Query input into AdditiveAttention should be a 2D tensor')

        self.attend_w = self.add_weight(shape=(context_shape[-1], query_shape[-1]), initializer=self.initializer,
                                        regularizer=self.regularizer, constraint=self.constraint,
                                        name='{}_attend_w'.format(self.name))
        super(MultiplicativeAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        context, query = inputs
        if mask is None:
            context_mask = None
        else:
            context_mask, _ = mask

        a = K.tanh(K.batch_dot(query, K.dot(context, self.attend_w), axes=[1, 2]))
        a = K.exp(a)
        if context_mask is not None:
            a *= K.cast(context_mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.sum(context * K.expand_dims(a), axis=1)

        if self.return_attend_weight:
            return attend_context, a
        else:
            return attend_context

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, _ = input_shape
        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]), (context_shape[0], context_shape[1])]
        else:
            return context_shape[0], context_shape[-1]


class DotProductAttention(Layer):
    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(DotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, query_shape = input_shape
        if len(context_shape) != 3:
            raise ValueError('Context input into AdditiveAttention should be a 3D tensor')
        if len(query_shape) != 2:
            raise ValueError('Query input into AdditiveAttention should be a 2D tensor')

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        context, query = inputs
        if mask is None:
            context_mask = None
        else:
            context_mask, _ = mask

        a = K.exp(K.batch_dot(query, context, axes=[1, 2]))

        # apply mask before normalization (softmax)
        if context_mask is not None:
            a *= K.cast(context_mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.sum(context * K.expand_dims(a), axis=1)

        if self.return_attend_weight:
            return attend_context, a
        else:
            return attend_context

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, _ = input_shape
        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]), (context_shape[0], context_shape[1])]
        else:
            return context_shape[0], context_shape[-1]


class ConcatAttention(Layer):
    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(ConcatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        context_shape, query_shape = input_shape
        if len(context_shape) != 3:
            raise ValueError('Context input into AdditiveAttention should be a 3D tensor')
        if len(query_shape) != 2:
            raise ValueError('Query input into AdditiveAttention should be a 2D tensor')
        concat_shape = context_shape[-1] + query_shape[-1]

        self.W = self.add_weight(shape=(concat_shape, concat_shape),  initializer=self.initializer,
                                 name='{}_W'.format(self.name), regularizer=self.regularizer, constraint=self.constraint)

        self.b = self.add_weight(shape=(concat_shape,), initializer='zero', name='{}_b'.format(self.name),
                                 regularizer=self.regularizer, constraint=self.constraint)

        self.u = self.add_weight(shape=(concat_shape, 1), initializer=self.initializer, name='{}_u'.format(self.name),
                                 regularizer=self.regularizer, constraint=self.constraint)

        super(ConcatAttention, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        context, query = inputs
        if mask is None:
            context_mask = None
        else:
            context_mask, _ = mask

        time_step = K.shape(context)[1]
        repeat_query = K.repeat(query, time_step)
        concat_input = K.concatenate([context, repeat_query])

        a = K.squeeze(K.dot(K.tanh(K.dot(concat_input, self.W) + self.b), self.u), axis=-1)
        a = K.exp(a)
        # apply mask before normalization (softmax)
        if context_mask is not None:
            a *= K.cast(context_mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.sum(context * K.expand_dims(a), axis=1)

        if self.return_attend_weight:
            return attend_context, a
        else:
            return attend_context

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, _ = input_shape
        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]), (context_shape[0], context_shape[1])]
        else:
            return context_shape[0], context_shape[-1]









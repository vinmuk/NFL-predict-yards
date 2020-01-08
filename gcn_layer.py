# these codes are from 
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D

def filter_dot(fltr, features):
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        return K.batch_dot(fltr, features)


class GraphConv(Layer):
    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel)
        output = filter_dot(fltr, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GraphAttention2(GraphConv):
    def __init__(self,
                 channels,
                 attn_heads=1,
                 attn_heads_reduction='average',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super().__init__(channels, **kwargs)
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.channels = channels
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.kernels2 = []       # Layer kernels for attention heads
        self.biases2 = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., attention_heads * channels)
            self.output_dim = self.channels * self.attn_heads
        else:
            # Output will have shape (..., channels)
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(input_dim, self.channels),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)
            kernel2 = self.add_weight(shape=(input_dim, self.channels),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels2.append(kernel2)
            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.channels,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)
            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.channels, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head))
            attn_kernel_neighs = self.add_weight(shape=(self.channels, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            kernel2 = self.kernels2[head]
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')
            features2 = K.dot(X, kernel2)
            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            if len(K.int_shape(features)) == 2:
                attn_for_neighs_T = K.transpose(attn_for_neighs)
            else:
                attn_for_neighs_T = K.permute_dimensions(attn_for_neighs, (0, 2, 1))
            dense = attn_for_self + attn_for_neighs_T

            # Add nonlinearity
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Convolution
            node_features = filter_dot(dropout_attn, dropout_feat)
            node_features += features2 
            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.output_dim,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GlobalPooling(Layer):
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.supports_masking = True
        self.pooling_op = None

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'graph':
            X = inputs[0]
            I = inputs[1]
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == 'graph':
            return self.pooling_op(X, I)
        else:
            return K.sum(X, axis=-2, keepdims=(self.data_mode == 'single'))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1,) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]  # Input shape is a list of shapes for X and I

    def get_config(self):
        return super().get_config()
    
class GlobalAttnSumPool(GlobalPooling):

    def __init__(self,
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
            F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
            F = input_shape[-1]
        # Attention kernels
        self.attn_kernel = self.add_weight(shape=(F, 1),
                                           initializer=self.attn_kernel_initializer,
                                           regularizer=self.attn_kernel_regularizer,
                                           constraint=self.attn_kernel_constraint,
                                           name='attn_kernel')

        self.built = True

    def call(self, inputs):
        if self.data_mode == 'graph':
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs
        attn_coeff = K.dot(X, self.attn_kernel)
        attn_coeff = K.squeeze(attn_coeff, -1)
        attn_coeff = K.softmax(attn_coeff)
        if self.data_mode == 'single':
            output = K.dot(attn_coeff[None, ...], X)
        elif self.data_mode == 'batch':
            output = K.batch_dot(attn_coeff, X)
        else:
            output = attn_coeff[:, None] * X
            output = tf.segment_sum(output, I)

        return output

    def get_config(self):
        config = {
            'attn_kernel_initializer': self.attn_kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'attn_kernel_constraint': self.attn_kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
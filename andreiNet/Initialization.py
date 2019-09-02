import numpy as np


# Weight Initialization

def init_layer_weight_he_norm(input_shape, output_shape):
    stdev = np.sqrt(2.0 / input_shape)
    return np.random.normal(scale=stdev, size=(input_shape, output_shape))


def init_layer_weight_unit_norm(input_shape, output_shape):
    return np.random.normal(size=(input_shape, output_shape))


def init_layer_weights_ones(input_shape, output_shape):
    return np.ones((input_shape, output_shape))


# Bias Initialization

def init_layer_bias_zeros(length):
    return np.zeros(length)


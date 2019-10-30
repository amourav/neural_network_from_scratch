import numpy as np


# Weight Initialization

def init_layer_weight_he_norm(input_shape, output_shape):
    """
    he norm weight initialization for a single layer
    :param input_shape: input shape - int
    :param output_shape: output shape - int
    :return: Initialized weights - npy arr
    """
    stdev = np.sqrt(2.0 / input_shape)
    return np.random.normal(scale=stdev, size=(input_shape, output_shape))


def init_layer_weight_unit_norm(input_shape, output_shape):
    """
    unit norm weight initialization for a single layer
    :param input_shape: input shape - int
    :param output_shape: output shape - int
    :return: Initialized weights - npy arr
    """
    return np.random.normal(size=(input_shape, output_shape))


def init_layer_weights_ones(input_shape, output_shape):
    """
    initialize layer to be an array of ones
    :param input_shape: input shape - int
    :param output_shape: output shape - int
    :return: Initialized weights - npy arr
    """
    return np.ones((input_shape, output_shape))


# Implemented Weight Initializations
implemented_weight_init_dict = {'unit_norm': init_layer_weight_unit_norm,
                                'ones': init_layer_weights_ones,
                                'he_norm': init_layer_weight_he_norm,
                                }


# Bias Initialization
def init_layer_bias_zeros(length):
    """
    Initialize bias values for a single layer
    :param length: layer length - int
    :return: Initialized bias values - npy arr
    """
    return np.zeros(length)

# Implemented Bias Initializations
implemented_bias_init_dict = {'zeros': init_layer_bias_zeros,
                              }


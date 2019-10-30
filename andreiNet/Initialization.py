import numpy as np


# Weight Initialization
class InitLayerWeight:  # Base Class
    def __init__(self):
        pass

    def get_array(self, input_shape, output_shape):
        raise NotImplementedError


class HeNorm(InitLayerWeight):
    def __init__(self):
        super().__init__()

    def get_array(self, input_shape, output_shape):
        """
        he norm weight initialization for a single layer
        :param input_shape: input shape - int
        :param output_shape: output shape - int
        :return: Initialized weights - npy arr
        """
        stdev = np.sqrt(2.0 / input_shape)
        return np.random.normal(scale=stdev, size=(input_shape, output_shape))


class UnitNorm(InitLayerWeight):
    def __init__(self):
        super().__init__()

    def get_array(self, input_shape, output_shape):
        """
        unit norm weight initialization for a single layer
        :param input_shape: input shape - int
        :param output_shape: output shape - int
        :return: Initialized weights - npy arr
        """
        return np.random.normal(size=(input_shape, output_shape))


class Ones(InitLayerWeight):
    def __init__(self):
        super().__init__()

    def get_array(self, input_shape, output_shape):
        """
        initialize layer to be an array of ones
        :param input_shape: input shape - int
        :param output_shape: output shape - int
        :return: Initialized weights - npy arr
        """
        return np.ones((input_shape, output_shape))


# Implemented Weight Initializations
implemented_weight_init_dict = {'unit_norm': UnitNorm,
                                'ones': Ones,
                                'he_norm': HeNorm,
                                }


# Bias Initialization
class InitLayerBiases:  # Base Class
    def __init__(self):
        pass

    def get_array(self, length):
        raise NotImplementedError


class Zeros(InitLayerBiases):
    def __init__(self):
        super().__init__()

    def get_array(self, length):
        """
        Initialize bias values for a single layer
        :param length: layer length - int
        :return: Initialized bias values - npy arr
        """
        return np.zeros(length)


# Implemented Bias Initializations
implemented_bias_init_dict = {'zeros': Zeros,
                              }


import numpy as np


def softmax(z):
    """
    softmax activation for last layer
    :param z: layer input - npy arr
    :return: softmax activation - npy arr
    """
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    exp = np.exp(z - np.max(z)) # + eps * np.ones(z.shape)
    return exp / np.sum(exp, axis=1)[:, None]


def softmax_gradient(z, sm=None):
    """
    gradient of softmax activation (da/dz)
    :param z: layer input - npy arr
    :param sm: optional layer output (a) - npy arr
    :return: da/dz - npy arr
    """
    # https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
    if sm is None:
        sm = softmax(z)
    res = np.einsum('ij,ik->ijk', sm, -sm)
    np.einsum('ijj->ij', res)[...] += sm
    return res


def linear(z):
    """
    intermetiate layer activation
    :param z: layer input - npy arr
    :return: z npy arr
    """
    return z


def linear_derivative(z):
    """
    Derivative of linear activation
    :param z: input - npy arr
    :return: da/dz - npy arr
    """
    return np.ones((z.shape[1]))


def linear_gradient(z):
    """
    Gradient of linear last activation
    :param z: layer input - npy arr
    :return: da/dz - npy arr
    """
    N, K = z.shape
    I = np.repeat(np.eye(K, K)[np.newaxis, :, :], N, axis=0)
    return I


def sigmoid(z):
    """
    Sigmoid intermediate layer activation
    :param z: layer input - npy arr
    :return: sigmoid grad - npy arr
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Sigmoid activation grad
    :param z: layer input - npy arr
    :return: da/dz - npy array
    """
    s = sigmoid(z)
    return s * (1 - z)


def ReLU(z):
    """
    rectified linear Unit activation
    :param z: layer input - npy arr
    :return: relu activation - npy arr
    """
    return np.maximum(z, 0, z)


def ReLU_derivative(z):
    """
    Relu grad
    :param z: layer input - npy arr
    :return: da/dz - npy arr
    """
    return (z > 0).astype(int)


# Implemented activations
implemented_activations_dict = {'sigmoid': sigmoid,
                                'ReLU': ReLU,
                                'linear': linear}

# Implemented activations grad
implemented_act_derivative_dict = {'sigmoid': sigmoid_derivative,
                                   'ReLU': ReLU_derivative,
                                   'linear': linear_derivative
                                   }


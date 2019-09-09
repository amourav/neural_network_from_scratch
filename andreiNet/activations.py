import numpy as np

eps = 1e-10


def softmax(z):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    exp = np.exp(z - np.max(z)) # + eps * np.ones(z.shape)
    return exp / np.sum(exp, axis=1)[:, None]


def softmax_gradient(z, sm=None):
    # https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
    if sm is None:
        sm = softmax(z)
    res = np.einsum('ij,ik->ijk', sm, -sm)
    np.einsum('ijj->ij', res)[...] += sm
    return res


def linear(z):
    return z


def linear_derivative(z):
    return np.ones((z.shape[0]))


def linear_gradient(z):
    return np.eye((z.shape[0]))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - z)


def ReLU(z):
    return np.maximum(z, 0, z)


def ReLU_derivative(z):
    return (z > 0).astype(int)


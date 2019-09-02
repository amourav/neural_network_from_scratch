import numpy as np


def cross_entropy(y_true, y_pred):
    N = len(y_true)
    return -np.sum(y_true * np.log(y_pred)) / N


def cross_entropy_derivative(y_true, y_pred):
    N = len(y_true)
    return -(y_true / y_pred) / N


import numpy as np


def accuracy(y_true, y_pred):
    """
    measure accuracy of predictions (y_pred) given true labels (y_true)
    :param y_true: true class labels (numpy array) - e.g. np.array([0, 1, 2, 1])
    :param y_pred: predicted class labels (numpy array) e.g. np.array([0, 2, 1, 1])
    :return: accuracy (float)
    """
    if len(y_true.shape) > 1:
        y_true = y_true.argmax(axis=1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)
    return np.sum(y_true == y_pred) / len(y_true)


import numpy as np
from andreiNet.losses import (implemented_loss_dict,
                              Loss, MSE, CrossEntropy,
                              FocalLoss)


class Metric:  # base class - ignore
    def __init__(self):
        pass

    def eval(self, y_true, y_pred):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__

    def __str__(self):
        return self.__class__


class Accuracy(Loss):
    def __init__(self):
        super().__init__()

    def eval(self, y_true, y_pred):
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

    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def grad(self, y_true, y_pred):
        raise NotImplementedError

    def __repr__(self):
        return 'accuracy'

    def __str__(self):
        return 'accuracy'


# Implemented metrics

implemented_metric_dict = {'accuracy': Accuracy,
                           }
implemented_metric_dict.update(implemented_loss_dict)

# Metric criteria
metric_criteria_dict = {'accuracy': 'max',
                        'cross_entropy': 'min',
                        'focal_loss': 'min',
                        'mse': 'min'}


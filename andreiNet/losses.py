import numpy as np


class Loss:  # base class
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def grad(self, y_true, y_pred):
        raise NotImplementedError


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y_true, y_pred):
        """
        cross entropy loss for classification tasks
        :param y_true: ground truth targets - npy arr
        :param y_pred: predicted targets - npy arr
        :return: evaluated cross entropy - float
        """
        N = len(y_true)
        return -np.sum(y_true * np.log(y_pred)) / N

    def grad(self, y_true, y_pred):
        """
        Derivative of cross entropy loss
        :param y_true: ground truth targets - npy arr
        :param y_pred: predicted targets - npy arr
        :return: gradient of loss - npy arr
        """
        N = len(y_true)
        return -(y_true / y_pred) / N

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y_true, y_pred):
        """
        Mean Squared Error loss
        :param y_true: ground truth targets - npy arr
        :param y_pred: predicted targets - npy arr
        :return: Evaluated loss - float
        """
        N = len(y_true)
        return np.sum(np.square(y_true - y_pred)) / N

    def grad(self, y_true, y_pred):
        """
        Derivative of loss function
        :param y_true: ground truth targets - npy arr
        :param y_pred: predicted targets - npy arr
        :return: grad of loss - npy arr
        """
        N = len(y_true)
        return (-2 / N) * (y_true - y_pred)

# Implemented loss functions
implemented_loss_dict = {'cross_entropy': CrossEntropy,
                         'MSE': MSE}


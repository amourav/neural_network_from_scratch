import numpy as np


class Loss:  # base class
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def grad(self, y_true, y_pred):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__

    def __str__(self):
        return self.__class__


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

    def __repr__(self):
        return 'cross_entropy'

    def __str__(self):
        return 'cross_entropy'


class FocalLoss(Loss):
    # https://arxiv.org/pdf/1708.02002.pdf
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        assert(gamma >= 0)
        self._set_alpha()

    def _set_alpha(self):
        """
        set class weight parameter (alpha)
        :return: alpha - 1.0 (float) or numpy array (n_classes)
        """
        if self.alpha is None:
            self.alpha = 1.0
        elif type(self.alpha) is int or type(self.alpha) is float:
            self.alpha = float(self.alpha)
        elif type(self.alpha) is list:
            self.alpha = np.array(self.alpha)
        else:
            raise Exception('alpha ({}) not accepted'.format(self.alpha))

    def loss(self, y_true, y_pred):
        """
        focal loss for classification tasks
        :param y_true: ground truth targets - npy arr
        :param y_pred: predicted targets - npy arr
        :return: evaluated cross entropy - float
        """

        # down weight easy examples
        weight = self.alpha * (1 - y_true * y_pred) ** self.gamma

        N = len(y_true)
        return -np.sum(y_true * weight * np.log(y_pred)) / N

    def grad(self, y_true, y_pred):
        """
        Derivative of focal loss
        https://www.wolframalpha.com/input/?i=-%281-x%29%5Ek*log%28x%29
        :param y_true: ground truth targets - npy arr
        :param y_pred: predicted targets - npy arr
        :return: gradient of loss - npy arr
        """
        tmp = 1 - y_pred
        N = len(y_true)
        tmp2 = self.alpha * y_true * tmp ** self.gamma / N
        return -tmp2 * (self.gamma * np.log(y_pred) / tmp + 1 / y_pred)

    def __repr__(self):
        return 'focal_loss'

    def __str__(self):
        return 'focal_loss'


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

    def __repr__(self):
        return 'mse'

    def __str__(self):
        return 'mse'


# Implemented loss functions
implemented_loss_dict = {'cross_entropy': CrossEntropy,
                         'mse': MSE}


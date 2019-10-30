import numpy as np


class Activation:
    def __init__(self):
        pass

    def act(self, z):
        raise NotImplementedError

    def derivative(self, z):
        # use derivative for intermediate layers
        raise NotImplementedError

    def grad(self, z):
        # use gradient for last layer only
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        return type(self).__name__


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def act(self, z):
        """
        softmax activation for last layer
        :param z: layer input - npy arr
        :return: softmax activation - npy arr
        """
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        exp = np.exp(z - np.max(z))  # + eps * np.ones(z.shape)
        return exp / np.sum(exp, axis=1)[:, None]

    def derivative(self, z):
        raise Exception('derivative not appropriate for Softmax, use grad instead')

    def grad(self, z, sm=None):
        """
        gradient of softmax activation (da/dz)
        :param z: layer input - npy arr
        :param sm: optional layer output (a) - npy arr
        :return: da/dz - npy arr
        """
        # https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
        if sm is None:
            sm = self.act(z)
        res = np.einsum('ij,ik->ijk', sm, -sm)
        np.einsum('ijj->ij', res)[...] += sm
        return res

    def __repr__(self):
        return 'softmax'

    def __str__(self):
        return 'softmax'


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def act(self, z):
        """
        intermetiate layer activation
        :param z: layer input - npy arr
        :return: z npy arr
        """
        return z

    def derivative(self, z):
        """
        Derivative of linear activation
        :param z: input - npy arr
        :return: da/dz - npy arr
        """
        return np.ones((z.shape[1]))

    def grad(self, z):
        """
        Gradient of linear last activation
        :param z: layer input - npy arr
        :return: da/dz - npy arr
        """
        N, K = z.shape
        I = np.repeat(np.eye(K, K)[np.newaxis, :, :], N, axis=0)
        return I

    def __repr__(self):
        return 'linear'

    def __str__(self):
        return 'linear'


class Sigmoid(Activation):
    def __init__(self):
        pass

    def act(self, z):
        """
        Sigmoid intermediate layer activation
        :param z: layer input - npy arr
        :return: sigmoid grad - npy arr
        """
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        """
        Sigmoid activation grad
        :param z: layer input - npy arr
        :return: da/dz - npy array
        """
        s = self.act(z)
        return s * (1 - z)

    def grad(self, z):
        raise Exception('grad not appropriate for non-last layer')

    def __repr__(self):
        return 'sigmoid'

    def __str__(self):
        return 'sigmoid'


class ReLU(Activation):
    def __init__(self):
        pass

    def act(self, z):
        """
        rectified linear Unit activation
        :param z: layer input - npy arr
        :return: relu activation - npy arr
        """
        return np.maximum(z, 0, z)

    def derivative(self, z):
        """
        Relu grad
        :param z: layer input - npy arr
        :return: da/dz - npy arr
        """
        return (z > 0).astype(int)

    def grad(self, z):
        raise Exception('grad not appropriate for intermediate layer (derivative instead)')

    def __repr__(self):
        return 'relu'

    def __str__(self):
        return 'relu'


# Implemented activations
implemented_activations_dict = {'sigmoid': Sigmoid,
                                'relu': ReLU,
                                'linear': Linear}



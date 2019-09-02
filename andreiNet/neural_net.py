import numpy as np


def init_layer_weight_he_norm(input_shape, output_shape):
    stdev = np.sqrt(2.0 / input_shape)
    return np.random.normal(scale=stdev, size=(input_shape, output_shape))


def init_layer_weight_unit_norm(input_shape, output_shape):
    return np.random.normal(size=(input_shape, output_shape))


def init_layer_weights_ones(input_shape, output_shape):
    return np.ones((input_shape, output_shape))


def init_layer_bias_zeros(length):
    return np.zeros(length)


def softmax(z):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1)[:, None]


def softmax_gradient(z, sm=None):
    # https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
    if sm is None:
        sm = softmax(z)
    res = np.einsum('ij,ik->ijk', sm, -sm)
    np.einsum('ijj->ij', res)[...] += sm
    return res


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
    # np.repeat(np.eye(K, K)[np.newaxis, :, :], N, axis=0).shape
    return np.ones((z.shape[0]))


def linear_gradient(z):
    # np.repeat(np.eye(K, K)[np.newaxis, :, :], N, axis=0).shape
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


def cross_entropy(y_true, y_pred):
    N = len(y_true)
    return -np.sum(y_true * np.log(y_pred)) / N


def cross_entropy_derivative(y_true, y_pred):
    N = len(y_true)
    return -(y_true / y_pred)  # / N


def one_hot_encode(y, n_classes):
    y_one_hot = np.zeros((len(y), n_classes))
    for i, y_i in enumerate(y):
        y_one_hot[i, y_i] = 1
    return y_one_hot


def normalize_trn_data(X):
    """
    normalize data to have zero mean and unit variance
    :param X: input data (array) - X.shape = (n_samples, m_features)
    :return:
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)


def shuffle_data(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def batch_iterator(X, y, batch_size):
    N, _ = X.shape
    batch_idxs = np.arange(0, N, batch_size)

    for start in batch_idxs:
        stop = start + batch_size
        X_batch, y_batch = X[start:stop], y[start:stop]
        yield X_batch, y_batch


class NeuralNetwork:

    def __init__(self,
                 hidden=(8, 6),
                 init_weights='he_norm',
                 init_bias='zeros',
                 activation='ReLU',
                 loss='cross_entropy',
                 mode='classification',
                 shuffle=True,
                 verbose=False,
                 batch_size=10,
                 random_state=1):
        self.hidden = hidden
        self.init_weights = init_weights
        self.init_bias = init_bias
        self.activation = activation
        self.loss = loss
        self.random_state = random_state
        self.mode = mode
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        np.random.seed(self.random_state)
        self._set_act_func()
        self._set_loss()

    def _init_neural_network(self):
        implemented_weight_inits = {'unit_norm': init_layer_weight_unit_norm,
                                    'ones': init_layer_weights_ones,
                                    'he_norm': init_layer_weight_he_norm,
                                    }
        implemented_bias_inits = {'zeros': init_layer_bias_zeros,
                                  }
        try:
            init_layer_weight = implemented_weight_inits[self.init_weights]
            init_layer_bias = implemented_bias_inits[self.init_bias]
        except KeyError:
            raise Exception('{} or {} not accepted'.format(self.init_weights,
                                                           self.init_bias))

        self.weights = []
        self.biases = []
        for layer in range(len(self.hidden) + 1):
            if layer == 0:
                input_shape = self.n_features
                output_shape = self.hidden[layer]
            elif layer == len(self.hidden):
                input_shape = self.hidden[layer - 1]
                output_shape = self.n_classes
            else:
                input_shape = self.hidden[layer - 1]
                output_shape = self.hidden[layer]
            w_l = init_layer_weight(input_shape, output_shape)
            b_l = init_layer_bias(output_shape)
            self.weights.append(w_l)
            self.biases.append(b_l)

    def _set_act_func(self):
        implemented_activations = {'sigmoid': sigmoid,
                                   'ReLU': ReLU,
                                   'linear': linear}
        # set activation function
        try:
            self.act = implemented_activations[self.activation]
        except KeyError:
            raise Exception('{} not accepted'.format(self.activation))

        implemented_derivatives = {'sigmoid': sigmoid_derivative,
                                   'ReLU': ReLU_derivative,
                                   'linear': linear_derivative}

        # set activation derivative (da/dz)
        try:
            self.act_derivative = implemented_derivatives[self.activation]
        except KeyError:
            raise Exception('derivative not implemented for {}'.format(self.activation))

        # set activation for last layer (softmax for classification and linear for regression)
        if self.mode == 'classification':
            self.last_act = softmax
            self.last_act_grad = softmax_gradient
        elif self.mode == 'regression':
            self.last_act = linear
            self.last_act_grad = linear_gradient
        else:
            raise Exception('{} not accepted.'.format(self.mode))

    def _set_loss(self):
        implemented_losses = {'cross_entropy': cross_entropy, }
        loss_gradients = {'cross_entropy': cross_entropy_derivative, }
        try:
            self.loss_func = implemented_losses[self.loss]
            self.loss_grad_func = loss_gradients[self.loss]
        except KeyError:
            raise Exception('{} not accepted'.format(self.loss))

    def train(self, X, y, n_epochs=10, lr=0.001, n_classes=None):
        self.n_samples, self.n_features = X.shape
        self.classes = n_classes
        if n_classes is None:
            self.classes = set(y)
            self.n_classes = len(self.classes)
        y_one_hot = one_hot_encode(y, self.n_classes)
        self._init_neural_network()

        for e in range(n_epochs):
            self.loss_e = 0
            # shuffle data

            if self.shuffle:
                X, y_one_hot = shuffle_data(X, y_one_hot)

            # iterate through batches
            for X_batch, y_batch in batch_iterator(X, y_one_hot, self.batch_size):
                self._feed_forward(X_batch)
                self._back_prop(X_batch, y_batch, lr)
                self.loss_batch = self.loss_func(y_batch, self.activations[-1])
                self.loss_e += self.loss_batch

        if self.verbose:
            print(e, 'trn loss = {}'.format(self.loss_e))
        print('epoch {}: final trn loss = {}'.format(e, self.loss_e))

    def _feed_forward(self, X):
        self.activations = []
        self.Z_list = []
        act = self.act
        for layer, (w_l, b_l) in enumerate(zip(self.weights, self.biases)):
            if layer == 0:
                prev = X
            else:
                prev = self.activations[-1]

            if layer == len(self.hidden):
                act = self.last_act
            Z_l = np.dot(prev, w_l) + b_l
            act_l = act(Z_l)
            self.activations.append(act_l)
            self.Z_list.append(Z_l)

    def predict(self, X):
        self._feed_forward(X)
        return self.activations[-1]

    def _get_gradient(self, y, a, z):
        # https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
        dL_da = self.loss_grad_func(y, a)
        da_dz = self.last_act_grad(z)
        return np.einsum('ij,ijk->ik', dL_da, da_dz)

    def _back_prop(self, X, y, lr):
        # gradient from last (output) layer
        self.dL_dz = self._get_gradient(y=y,
                                        a=self.activations[-1],
                                        z=self.Z_list[-1])

        new_weights, new_biases = [], []
        L = len(self.activations)
        for layer in range(L - 1, -1, -1):
            w_l, b_l = self.weights[layer], self.biases[layer]
            Z_l = self.Z_list[layer]
            # activation from previous layer
            if layer == 0:
                act_prev = X
            else:
                act_prev = self.activations[layer - 1]
            # layer gradient
            if layer < L - 1:
                dL_da = self.dL_dz @ self.weights[layer + 1].T  # dL_da wrt activation of current layer
                da_dz = self.act_derivative(Z_l)
                self.dL_dz = np.multiply(da_dz, dL_da)

            dL_dW = act_prev.T @ self.dL_dz  # Weight Error
            dL_db = np.sum(self.dL_dz, axis=0)  # Bias Error

            w_l -= lr * dL_dW  # update weights
            b_l -= lr * dL_db  # update biases
            new_weights.append(w_l)
            new_biases.append(b_l)

        self.weights = new_weights[::-1]
        self.biases = new_biases[::-1]


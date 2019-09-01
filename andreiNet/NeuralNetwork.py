import numpy as np


def weight_init_unit_norm(input_shape, output_shape):
    return np.random.normal(size=(input_shape, output_shape))


def weight_init_ones(input_shape, output_shape):
    return np.ones((input_shape, output_shape))


def bias_init_zeros(length):
    return np.zeros(length)


def softmax(x):
    # activation (a)
    # input: N x K array
    # output: N x K array
    # source: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=1)[:, None]


def softmax_derivative(z):
    # da/dz
    #input: N x K array
    #output: N x K x K array
    #http://saitcelebi.com/tut/output/part2.html
    N, K = z.shape
    s = softmax(z)[:, :, np.newaxis]
    a = np.tensordot(s, np.ones((1, K)), axes=([-1], [0]))
    I = np.repeat(np.eye(K, K)[np.newaxis, :, :], N, axis=0)
    b = I - np.tensordot(np.ones((K, 1)), s.T, axes=([-1], [0])).T
    return a * np.swapaxes(b, 1, 2)


def get_dE_dz(dE_da, da_dz):
    # array (N x K)
    # array (N x K x K)
    # output: array (N x K)
    N, K = dE_da.shape
    dE_dz = np.zeros((N, K))
    for n in range(N):
        dE_dz[n, :] = np.matmul(da_dz[n], dE_da[n, :, np.newaxis]).T
    return dE_dz


def linear(z):
    return z


def linear_derivative(z):
    N, K = z.shape
    I = np.repeat(np.eye(K, K)[np.newaxis, :, :], N, axis=0).shape
    return I


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def ReLU(x):
    return np.maximum(x, 0, x)


def ReLU_derivative(x):
    return (x > 0).astype(int)


def cross_entropy(y_true, y_pred):
    N = len(y_true)
    return -np.sum(y_true * np.log(y_pred)) / N


def cross_entropy_derivative(y_true, y_pred):
    N = len(y_true)
    return -(y_true / y_pred) / N


def one_hot_encode(y, n_classes):
    y_onehot = np.zeros((len(y), n_classes))
    for i, y_i in enumerate(y):
        y_onehot[i, y_i] = 1
    return y_onehot


class NeuralNetwork():

    def __init__(self,
                 hidden=(8, 6),
                 init_weights='unit_norm',
                 init_bias='zeros',
                 activation='sigmoid',
                 loss='cross_entropy',
                 mode='classification',
                 random_state=1):
        self.hidden = hidden
        self.init_weights = init_weights
        self.init_bias = init_bias
        self.activation = activation
        self.loss = loss
        self.random_state = random_state
        self.mode = mode
        np.random.seed(self.random_state)
        self._set_act_func()
        self._set_loss()

    def _init_neural_network(self):
        implemented_weight_inits = {'unit_norm': weight_init_unit_norm,
                                    'ones': weight_init_ones
                                    }
        implemented_bias_inits = {'zeros': bias_init_zeros,
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
                                   'linear': linear_derivative,
                                   'softmax': softmax}
        # set activation function
        try:
            self.act = implemented_activations[self.activation]
        except KeyError:
            raise Exception('{} not accepted'.format(self.activation))

        implemented_derivatives = {'sigmoid': sigmoid_derivative,
                                   'ReLU': ReLU_derivative,
                                   'linear': linear_derivative,
                                   'softmax': softmax_derivative}

        # set activation derivative (da/dz)
        try:
            self.act_derivative = implemented_derivatives[self.activation]
        except KeyError:
            raise Exception('derivative not implemented for {}'.format(self.activation))

        # set activation for last layer (softmax for classification and linear for regression)
        if self.mode == 'classification':
            self.last_act = softmax
            self.last_act_grad = softmax_derivative
        elif self.mode == 'regression':
            self.last_act = linear

    def _set_loss(self):
        implemented_losses = {'cross_entropy': cross_entropy, }
        try:
            self.loss_func = implemented_losses[self.loss]
        except KeyError:
            raise Exception('{} not accepted'.format(self.loss))

    def train(self, X, y, n_epochs=10, lr=0.001, n_classes=None):
        self.lr = lr
        self.n_samples, self.n_features = X.shape
        self.classes = n_classes
        if n_classes is None:
            self.classes = set(y)
            self.n_classes = len(self.classes)

        y_one_hot = one_hot_encode(y, self.n_classes)
        self._init_neural_network()

        print(self.biases[1])
        for e in range(n_epochs):
            # implement shuffle
            # implement batch
            self._feed_forward(X)
            self.loss_e = self.loss_func(y_one_hot, self.activations[-1])
            print('loss', e, self.loss_e)
            self._back_prop(X, y_one_hot)

        print(self.biases[1])

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

    def _dE_dZ(self, y, p):
        # dE/dz where E(y) - cross entropy and a(z) is the softmax activation function
        return p - y

    def _get_grad(dE_da, da_dz):
        return np.tensordot(dE_da, da_dz, axes=([-1], [0]))

    def _back_prop(self, X, y):
        y_pred = self.activations[-1]
        self.dE_da =
        self.da_dz =

        new_weights, new_biases = [], []
        L = len(self.activations)
        for layer in range(L - 1, -1, -1):
            w_l, b_l = self.weights[layer], self.biases[layer]
            Z_l = self.Z_list[layer]

            if layer == 0:
                act_prev = X
            else:
                act_prev = self.activations[layer - 1]

            if layer == L - 1:
                self.dE_dz = self._dE_dZ(y, y_pred)
            else:
                dE_da = self.dE_dz @ self.weights[layer + 1].T  # dE_da wrt activation of current layer
                da_dz = self.act_derivative(Z_l)
                self.dE_dz = np.multiply(da_dz, dE_da)

            dE_dW = act_prev.T @ self.dE_dz
            dE_db = np.sum(self.dE_dz, axis=0)
            # print(layer, act_prev.T.shape, self.dE_dz.shape, dE_dW.shape, w_l.shape)
            w_l -= self.lr * dE_dW
            b_l -= self.lr * dE_db

            new_weights.append(w_l)
            new_biases.append(b_l)

        self.weights = new_weights[::-1]
        self.biases = new_biases[::-1]
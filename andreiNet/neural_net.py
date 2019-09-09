import numpy as np
import copy
import sys

from andreiNet.utils import one_hot_encode, norm_data, shuffle_data, batch_iterator
from andreiNet.metrics import accuracy
from andreiNet.losses import cross_entropy, cross_entropy_derivative
from andreiNet.activations import (softmax, softmax_gradient,
                                   linear, linear_derivative, linear_gradient,
                                   sigmoid, sigmoid_derivative,
                                   ReLU, ReLU_derivative)
from andreiNet.Initialization import (init_layer_weight_he_norm,
                                      init_layer_weight_unit_norm,
                                      init_layer_weights_ones,
                                      init_layer_bias_zeros)


implemented_weight_inits = {'unit_norm': init_layer_weight_unit_norm,
                            'ones': init_layer_weights_ones,
                            'he_norm': init_layer_weight_he_norm,
                            }
implemented_bias_inits = {'zeros': init_layer_bias_zeros,
                          }
implemented_activations = {'sigmoid': sigmoid,
                           'ReLU': ReLU,
                           'linear': linear}
implemented_act_derivatives = {'sigmoid': sigmoid_derivative,
                               'ReLU': ReLU_derivative,
                               'linear': linear_derivative
                               }
implemented_losses = {'cross_entropy': cross_entropy, }
implemented_loss_gradients = {'cross_entropy': cross_entropy_derivative, }
implemented_metrics = {'accuracy': accuracy,
                       'cross_entropy': cross_entropy, }
metric_criteria = {'accuracy': 'max',
                   'cross_entropy': 'min', }

class NeuralNetwork:
    def __init__(self,
                 hidden=(8, 6),
                 init_weights='he_norm',
                 init_bias='zeros',
                 activation='ReLU',
                 loss='cross_entropy',
                 mode='classification',
                 metrics=None,
                 shuffle=True,
                 verbose=False,
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
        self.metrics = metrics

    def _set_weight_init(self):
        try:
            self.init_layer_weight = implemented_weight_inits[self.init_weights]
        except KeyError:
            raise Exception('{} not accepted'.format(self.init_weights))

    def _set_bias_init(self):
        try:
            self.init_layer_bias = implemented_bias_inits[self.init_bias]
        except KeyError:
            raise Exception('{} not accepted'.format(self.init_bias))

    def _init_history(self):
        self.metrics.append(self.loss)
        self.trn_metric_hist = {}
        self.val_metric_hist = {}
        if self.metrics is not None:
            for metric in self.metrics:
                self.trn_metric_hist[metric] = []
                self.val_metric_hist[metric] = []

    def _init_neural_network(self):
        np.random.seed(self.random_state)
        self._set_act_func()
        self._set_loss()
        self._set_weight_init()
        self._set_bias_init()
        self._init_history()
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
            w_l = self.init_layer_weight(input_shape,
                                         output_shape)
            b_l = self.init_layer_bias(output_shape)
            self.weights.append(w_l)
            self.biases.append(b_l)

    def _set_act_func(self):
        # set activation function
        try:
            self.act = implemented_activations[self.activation]
        except KeyError:
            raise Exception('{} not accepted'.format(self.activation))

        # set activation derivative (da/dz)
        try:
            self.act_derivative = implemented_act_derivatives[self.activation]
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
        try:
            self.loss_func = implemented_losses[self.loss]
            self.loss_grad_func = implemented_loss_gradients[self.loss]
        except KeyError:
            raise Exception('{} not accepted'.format(self.loss))

    def _encode(self, y, n_classes):
        self.n_classes = n_classes
        if n_classes is None and self.mode == 'classification':
            self.n_classes = len(set(y))
        y_one_hot = one_hot_encode(y, self.n_classes)
        return y_one_hot

    def _set_early_stop(self, early_stop):
        self.early_stop = early_stop
        if early_stop is not None:
            self.stop_metric, self.patience = self.early_stop
            if self.stop_metric not in self.metrics:
                self.metrics.append(self.stop_metric)

    def _get_metrics(self, X, y):
        metric_vals = {}
        if self.metrics is None:
            return metric_vals
        y_pred = self.predict(X)
        for metric in self.metrics:
            try:
                metric_func = implemented_metrics[metric]
            except KeyError:
                raise Exception('{} not accepted metric'.format(metric))
            metric_vals[metric] = metric_func(y, y_pred)
        return metric_vals

    def _update_metrics(self, X, y_ohe, val_data):
        self.metrics_trn = self._get_metrics(X, y_ohe)
        if val_data is not None:
            X_val, y_val = val_data
            y_val_ohe = self._encode(y_val, self.n_classes)
            self.metrics_val = self._get_metrics(X_val, y_val_ohe)

    def _update_history(self, update_val=False):
        for metric in self.metrics:
            self.trn_metric_hist[metric].append(self.metrics_trn[metric])
            if update_val:
                self.val_metric_hist[metric].append(self.metrics_val[metric])

    def _get_stop_criteria(self, epoch):
        stop_criteria = False
        if self.early_stop is None:
            return stop_criteria
        f = -1 if metric_criteria[self.stop_metric] == 'max' else 1
        current_score = f * self.val_metric_hist[self.stop_metric][-1]
        if epoch == 1:
            self.counter = 0
            self.best_score = current_score
        else:
            if current_score < self.best_score:
                self.counter = 0
                self.best_score = current_score
            else:
                self.counter += 1

        if self.counter > self.patience:
            stop_criteria = True

        if stop_criteria:
            print("early stop: epoch {} patience {}".format(epoch,
                                                            self.patience))
        return stop_criteria

    def _model_checkpoint(self, current_score, epoch):
        self.best_val_loss = current_score
        self.best_model = (copy.deepcopy(self.weights),
                           copy.deepcopy(self.biases),
                           epoch)

    def _update_best_model(self, epoch, save_best, val_data):
        if save_best and (val_data is not None):
            current_score = self.val_metric_hist[self.loss][-1]
            if epoch == 1:
                print('model checkpoint {}'.format(epoch))
                self._model_checkpoint(current_score, epoch)
            else:
                if current_score < self.best_val_loss:
                    print('model checkpoint {}'.format(epoch))
                    self.best_val_loss = current_score
                    self._model_checkpoint(current_score, epoch)

    def _set_best_model(self, save_best):
        if save_best:
            print('setting best model from epoch {}'.format(self.best_model[2]))
            self.weights = self.best_model[0]
            self.biases = self.best_model[1]

    def _get_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss_func(y, y_pred)

    def train(self, X, y,
              val_data=None,
              n_epochs=10, lr=0.001,
              batch_size=10,
              n_classes=None,
              save_best=False,
              early_stop=None):

        self._set_early_stop(early_stop)
        self.n_samples, self.n_features = X.shape
        #n_batches = np.ceil(self.n_samples / batch_size)
        y_one_hot = self._encode(y, n_classes)

        self._init_neural_network()
        for e in range(1, n_epochs + 1):
            self.loss_e = 0
            # shuffle data
            if self.shuffle:
                X, y_one_hot = shuffle_data(X, y_one_hot)
            # iterate through batches
            for X_batch, y_batch in batch_iterator(X, y_one_hot, batch_size):
                self._feed_forward(X_batch)
                self._back_prop(X_batch, y_batch, lr)
                #self.loss_batch = self.loss_func(y_batch, self.activations[-1])
                #self.loss_e += self.loss_batch / n_batches
            self.loss_e = self._get_loss(X, y_one_hot)
            self._update_metrics(X, y_one_hot, val_data)
            self._update_history(update_val=(val_data is not None))
            self._update_best_model(e, save_best, val_data)
            stop_criteria = self._get_stop_criteria(e)
            if stop_criteria:
                break
            if self.verbose or e == n_epochs:
                print('epoch {}: final trn loss = {} trn metrics {}'.format(e,
                                                                            self.loss_e,
                                                                            self.metrics_trn))
                if val_data is not None:
                    print('val metrics {}'.format(self.metrics_val))
        self._set_best_model(save_best)

    def _feed_forward(self, X):
        self.activations = []
        self.z_list = []
        act = self.act
        for layer, (w_l, b_l) in enumerate(zip(self.weights, self.biases)):
            if layer == 0:
                prev = X
            else:
                prev = self.activations[-1]
            if layer == len(self.hidden):
                act = self.last_act
            z_l = np.dot(prev, w_l) + b_l
            act_l = act(z_l)
            self.activations.append(act_l)
            self.z_list.append(z_l)

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
                                        z=self.z_list[-1])
        new_weights, new_biases = [], []
        L = len(self.activations)
        for layer in range(L - 1, -1, -1):
            w_l, b_l = self.weights[layer], self.biases[layer]
            z_l = self.z_list[layer]
            # activation from previous layer
            if layer == 0:
                act_prev = X
            else:
                act_prev = self.activations[layer - 1]
            # layer gradient
            if layer < L - 1:
                dL_da = self.dL_dz @ self.weights[layer + 1].T  # dL_da wrt activation of current layer
                da_dz = self.act_derivative(z_l)
                self.dL_dz = np.multiply(da_dz, dL_da)

            dL_dW = act_prev.T @ self.dL_dz  # Weight Error
            dL_db = np.sum(self.dL_dz, axis=0)  # Bias Error

            w_l -= lr * dL_dW  # update weights
            b_l -= lr * dL_db  # update biases
            new_weights.append(w_l)
            new_biases.append(b_l)

        self.weights = new_weights[::-1]
        self.biases = new_biases[::-1]


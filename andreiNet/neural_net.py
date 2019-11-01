import numpy as np
import copy

from andreiNet.utils import one_hot_encode, shuffle_data, batch_iterator, get_instance
from andreiNet.metrics import (Metric, Accuracy,
                               implemented_metric_dict,
                               metric_criteria_dict)
from andreiNet.losses import (implemented_loss_dict,
                              Loss, CrossEntropy, MSE)
from andreiNet.activations import (implemented_activations_dict,
                                   Activation, Softmax, Linear,
                                   Sigmoid, ReLU)
from andreiNet.Initialization import (implemented_weight_init_dict,
                                      InitLayerWeight,
                                      HeNorm, UnitNorm, Ones,
                                      implemented_bias_init_dict,
                                      InitLayerBiases, Zeros)


class NeuralNetwork:
    """
    neural network implementation from scratch (ok just numpy) with
    vectorized forward and back prop.
    Example:
    from andreiNet.neural_net import NeuralNetwork
    nn = NeuralNetwork.init()

    """

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
        """
        Set the weight initialization procedure or throw error if not implemented
        """
        weight_init = get_instance(self.init_weights,
                                   implemented_weight_init_dict,
                                   InitLayerWeight,
                                   error_msg='weight init error')
        self.init_layer_weight = weight_init.get_array

    def _set_bias_init(self):
        """
        Set the bias initialization procedure or throw error if not implemented
        """
        bias_init = get_instance(self.init_bias,
                                 implemented_bias_init_dict,
                                 InitLayerBiases,
                                 error_msg='bias init error')
        self.init_layer_bias = bias_init.get_array

    def _init_history(self):
        """
        Initialize model history
        :return:
        """
        self.metrics.append(str(self.loss))
        self.trn_metric_hist = {}
        self.val_metric_hist = {}
        if self.metrics is not None:
            for metric in self.metrics:
                self.trn_metric_hist[metric] = []
                self.val_metric_hist[metric] = []

    def _init_neural_network(self):
        """
        Initialize model weights and biases
        """
        np.random.seed(self.random_state)
        self._set_act_func()
        self._set_loss()
        self._set_weight_init()
        self._set_bias_init()
        self._init_history()
        self._set_metrics()
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
        """
        Set the activation functions and respective gradients
        or throw error if not implemented
        """
        # set activation function
        activation = get_instance(self.activation,
                                  implemented_activations_dict,
                                  Activation,
                                  error_msg='activation function error')
        try:
            self.act = activation.act
            self.act_derivative = activation.derivative
        except KeyError:
            raise Exception('{} not accepted'.format(self.activation))

        # set activation for last layer (softmax for classification and linear for regression)
        if self.mode == 'classification':
            last_activation = Softmax()
        elif self.mode == 'regression':
            last_activation = Linear()
        else:
            raise Exception('{} not accepted.'.format(self.mode))
        self.last_act = last_activation.act
        self.last_act_grad = last_activation.grad

    def _set_loss(self):
        """
        Set loss function and gradient
        or throw error if not implemented
        """
        # init loss
        loss = get_instance(self.loss,
                            implemented_loss_dict,
                            Loss,
                            error_msg='Loss not accepted')
        self.loss_func = loss.loss
        self.loss_grad_func = loss.grad

    def _set_metrics(self):
        """
        set metric functions
        or throw error if not implemented
        """
        metric_functions = []
        for metric in self.metrics:
            metric_func = get_instance(metric,
                                       implemented_metric_dict,
                                       Loss,
                                       error_msg='metric not accepted').eval
            metric_functions.append(metric_func)
        self.metric_functions = metric_functions

    def _encode(self, y, n_classes):
        """
        One hot encode targets if in classification mode
        or encode a signle class for classification
        :param y: targets
        :param n_classes: number of classes
        :return: encoded target
        """
        self.n_classes = n_classes
        if self.mode == 'classification':
            if n_classes is None:
                self.n_classes = len(set(y))
            y = one_hot_encode(y, self.n_classes)
        elif self.mode == 'regression':
            y = y[:, None]
            self.n_classes = 1
        return y

    def _set_early_stop(self, early_stop):
        """
        Set early stopping criteria
        :param early_stop: stop criteria (metric, patience) - tuple
        """
        self.early_stop = early_stop
        if early_stop is not None:
            self.stop_metric, self.patience = self.early_stop
            if self.stop_metric not in self.metrics:
                self.metrics.append(self.stop_metric)

    def _get_metric_values(self, X, y):
        """
        Evaluate tracked metrics
        :param X: Input data - npy array
        :param y: targets - npy array
        :return: metric values - Dict
        """
        metric_vals = {}
        if self.metrics is None:
            return metric_vals
        y_pred = self.predict(X)
        for metric, metric_func in zip(self.metrics, self.metric_functions):
            metric_vals[metric] = metric_func(y, y_pred)
        return metric_vals

    def _update_metrics(self, X, y_ohe, val_data):
        """
        Evaluate tracked metrics for training
        and val data (optional)
        :param X: Input data - npy array
        :param y_ohe: targets - npy array
        :param val_data: validation data (X_val, y_val) - tuple
        """
        self.metrics_trn = self._get_metric_values(X, y_ohe)
        if val_data is not None:
            X_val, y_val = val_data
            y_val_ohe = self._encode(y_val, self.n_classes)
            self.metrics_val = self._get_metric_values(X_val, y_val_ohe)

    def _update_history(self, update_val=False):
        """
        Update model history
        :param update_val: update validation data - bool
        :return:
        """
        for metric in self.metrics:
            self.trn_metric_hist[metric].append(self.metrics_trn[metric])
            if update_val:
                self.val_metric_hist[metric].append(self.metrics_val[metric])

    def _get_stop_criteria(self, epoch):
        """
        Deterimine if stop criteria are met
        :param epoch: current epoch - int
        :return: stop criteria are met - bool
        """
        stop_criteria = False
        if self.early_stop is None:
            return stop_criteria
        f = -1 if metric_criteria_dict[self.stop_metric] == 'max' else 1
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
        """
        Update best model
        :param current_score: score on current epoch - float
        :param epoch: current epoch - int
        """
        self.best_val_loss = current_score
        self.best_model = (copy.deepcopy(self.weights),
                           copy.deepcopy(self.biases),
                           epoch)

    def _update_best_model(self, epoch, save_best, val_data):
        """
        Track best model on val data if save_best is true
        :param epoch: current epoch - int
        :param save_best: save best model (yes if true) - bool
        :param val_data: validation data (x_val, y_val) - tuple
        """
        if save_best and (val_data is not None):
            current_score = self.val_metric_hist[str(self.loss)][-1]
            if epoch == 1:
                if self.verbose:
                    print('model checkpoint {}'.format(epoch))
                self._model_checkpoint(current_score, epoch)
            else:
                if current_score < self.best_val_loss:
                    if self.verbose:
                        print('model checkpoint {}'.format(epoch))
                    self.best_val_loss = current_score
                    self._model_checkpoint(current_score, epoch)

    def _set_best_model(self, save_best):
        """
        Update weights to reflect the best model
        :param save_best: set best weights if true - bool
        """
        if save_best:
            print('setting best model from epoch {}'.format(self.best_model[2]))
            self.weights = self.best_model[0]
            self.biases = self.best_model[1]

    def _get_loss(self, X, y):
        """
        Evaludate loss
        :param X: Input data - npy arr
        :param y: target data - npy arr
        :return: loss value - float
        """
        y_pred = self.predict(X)
        return self.loss_func(y, y_pred)

    def train(self, X, y,
              val_data=None,
              n_epochs=10, lr=0.001,
              batch_size=10,
              n_classes=None,
              save_best=False,
              early_stop=None):
        """
        Train neural network

        :param X: training data - npy arr (n_samples, m_featuers)
        :param y: training targets - npy arr (n_samples, )
        :param val_data: validation data (X_val, y_val) - tuple
        :param n_epochs: number of training epochs - int
        :param lr: learning rate - float
        :param batch_size: samples per batch - int
        :param n_classes: number of classes (X.shape[1] by default) - int
        :param save_best: track best model on val data if true - bool
        :param early_stop: early stop criteria (stop metric, patience) - tuple
        """
        self._set_early_stop(early_stop)
        self.n_samples, self.n_features = X.shape
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
        """
        Forward propogate data and update weights
        :param X: data - npy array
        """
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
        """
        Predict targets from data
        :param X: input data - npy array
        :return: predictions - npy array
        """
        self._feed_forward(X)
        return self.activations[-1]

    def _get_gradient(self, y, a, z):
        """
        Gradient of loss w.r.t input to the last layer (Z_last)
        :param y: target - npy arr
        :param a: last layer activation (npy arr)
        :param z: last layer input (npy arr)
        :return: gradient (dL/dz) - array
        """
        # https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
        dL_da = self.loss_grad_func(y, a)
        da_dz = self.last_act_grad(z)
        return np.einsum('ij,ijk->ik', dL_da, da_dz)

    def _back_prop(self, X, y, lr):
        """
        Layer by layer back prop
        :param X: Input data
        :param y: target
        :param lr: learning rate
        """
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

            # Future work: implement custom optimizer
            w_l -= lr * dL_dW  # update weights
            b_l -= lr * dL_db  # update biases
            new_weights.append(w_l)
            new_biases.append(b_l)

        self.weights = new_weights[::-1]
        self.biases = new_biases[::-1]


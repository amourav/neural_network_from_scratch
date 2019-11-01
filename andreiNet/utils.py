import numpy as np


def one_hot_encode(y, n_classes):
    """
    one hot encode
    :param y: target data - npy arr
    :param n_classes: number of classes - int
    :return: encoded target - npy arr
    """
    y_ohe = np.zeros((len(y), n_classes))
    y_ohe[np.arange(len(y)), y] = 1
    return y_ohe


def norm_data(X):
    """
    normalize data to have zero mean and unit variance
    :param X: input data (array) - X.shape = (n_samples, m_features)
    :return: normalized data
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)


def shuffle_data(X, y):
    """
    Shuffle input and targets together
    :param X: input data - npy arr
    :param y: target data - npy arr
    :return: shuffled data (X, y) - tuple
    """
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def batch_iterator(X, y, batch_size):
    """
    Iterate of data to generate batches
    :param X: input data - npy arr
    :param y: target data - npy arr
    :param batch_size: batch size - int
    :yield: batch of input and target data
    """
    N, _ = X.shape
    batch_idxs = np.arange(0, N, batch_size)

    for start in batch_idxs:
        stop = start + batch_size
        X_batch, y_batch = X[start:stop], y[start:stop]
        yield X_batch, y_batch


def get_with_keyword(key_word, implemented_dict):
    """
    Check if method is class is implemented
    :param key_word: keyword parameter (str)
    :param implemented_dict: maps keyword to function or class (dict)
    :return: class or function
    """
    try:
        var = implemented_dict[key_word]
        return var
    except KeyError:
        raise Exception('{} not accepted'.format(key_word))


def get_instance(arg, implemented_dict, base_class, error_msg='error'):
    if type(arg) is str:
        instance = get_with_keyword(arg, implemented_dict)()
    elif isinstance(arg, base_class):
        instance = arg
    else:
        raise Exception(error_msg + ': ' + str(arg))
    return instance

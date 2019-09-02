import numpy as np


def one_hot_encode(y, n_classes):
    y_ohe = np.zeros((len(y), n_classes))
    y_ohe[np.arange(len(y)), y] = 1
    return y_ohe


def norm_data(X):
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


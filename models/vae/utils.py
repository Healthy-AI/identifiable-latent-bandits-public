import numpy as np
#from numba import njit
from sklearn.utils.validation import check_array


def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.

    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_helper(X, Y)


#@njit
def _pairwise_distances_no_broadcast_helper(X, Y):  # pragma: no cover
    """Internal function for calculating the distance with numba. Do not use.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Intermediate results. Do not use.

    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
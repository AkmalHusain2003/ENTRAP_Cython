# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport log

from entrap._core import compute_twenn_mu

cnp.import_array()


def estimate_intrinsic_dimension_twenn(cnp.ndarray X, bint X_is_dist=False):
    cdef int N = X.shape[0]
    cdef int i
    cdef double d_hat, max_dim

    if N < 3:
        return 1.0

    if X_is_dist:
        dist = np.ascontiguousarray(X, dtype=np.float64)
    else:
        from scipy.spatial.distance import squareform, pdist
        dist = np.ascontiguousarray(
            squareform(pdist(X, metric='euclidean')), dtype=np.float64
        )

    mu = compute_twenn_mu(dist)

    cdef cnp.ndarray[double, ndim=1] mu_arr = np.asarray(mu, dtype=np.float64)
    cdef cnp.ndarray[long, ndim=1] sort_idx = np.argsort(mu_arr)

    cdef cnp.ndarray[double, ndim=1] log_mu = np.log(mu_arr[sort_idx] + 1e-12)
    cdef cnp.ndarray[double, ndim=1] Femp = np.arange(N, dtype=np.float64) / N
    cdef cnp.ndarray[double, ndim=1] neg_log_F = np.empty(N, dtype=np.float64)

    cdef double[::1] neg_log_F_v = neg_log_F
    cdef double[::1] Femp_v = Femp

    with nogil:
        for i in range(N):
            neg_log_F_v[i] = -log(1.0 - Femp_v[i] + 1e-12)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression(fit_intercept=False)
    lr.fit(log_mu.reshape(-1, 1), neg_log_F.reshape(-1, 1))
    d_hat = lr.coef_[0][0]

    if X_is_dist:
        max_dim = float(N)
    else:
        max_dim = float(X.shape[1])

    if d_hat < 1.0:
        return 1.0
    if d_hat > max_dim:
        return max_dim
    return d_hat

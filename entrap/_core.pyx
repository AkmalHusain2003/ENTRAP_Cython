# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, exp, log, fabs
from libc.float cimport DBL_MAX

cnp.import_array()


cdef double _cov_single_row(const double* row, int m) nogil:
    cdef int j
    cdef double s = 0.0, mean, vs = 0.0, d
    for j in range(m):
        s += row[j]
    if m == 0:
        return DBL_MAX
    mean = s / m
    if mean > 1e-12:
        for j in range(m):
            d = row[j] - mean
            vs += d * d
        return sqrt(vs / m) / mean
    return DBL_MAX


def compute_cov_from_rows(cnp.ndarray[double, ndim=2] nd):
    cdef int n = nd.shape[0]
    cdef int m = nd.shape[1]
    cdef int i
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef double[:,::1] nd_v = np.ascontiguousarray(nd, dtype=np.float64)
    with nogil:
        for i in range(n):
            out_v[i] = _cov_single_row(&nd_v[i, 0], m)
    return out


def compute_cluster_mean(cnp.ndarray[double, ndim=2] pts):
    cdef int n = pts.shape[0]
    cdef int d = pts.shape[1]
    cdef int i, j
    cdef cnp.ndarray[double, ndim=1] out = np.zeros(d, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef double[:,::1] pts_v = np.ascontiguousarray(pts, dtype=np.float64)
    with nogil:
        for i in range(n):
            for j in range(d):
                out_v[j] += pts_v[i, j]
        for j in range(d):
            out_v[j] /= n
    return out


def compute_cluster_covariance(cnp.ndarray[double, ndim=2] pts,
                                cnp.ndarray[double, ndim=1] mean,
                                double ridge):
    cdef int n = pts.shape[0]
    cdef int d = pts.shape[1]
    cdef int i, j, k
    cdef double dj
    cdef cnp.ndarray[double, ndim=2] out = np.zeros((d, d), dtype=np.float64)
    cdef double[:,::1] out_v = out
    cdef double[:,::1] pts_v = np.ascontiguousarray(pts, dtype=np.float64)
    cdef double[::1] mean_v = np.ascontiguousarray(mean, dtype=np.float64)
    with nogil:
        for i in range(n):
            for j in range(d):
                dj = pts_v[i, j] - mean_v[j]
                for k in range(d):
                    out_v[j, k] += dj * (pts_v[i, k] - mean_v[k])
        for j in range(d):
            for k in range(d):
                out_v[j, k] /= n
            out_v[j, j] += ridge
    return out


def compute_mahalanobis_sq(cnp.ndarray[double, ndim=1] diff,
                            cnp.ndarray[double, ndim=2] Sigma_inv):
    cdef int d = diff.shape[0]
    cdef int i, j
    cdef double result = 0.0, temp
    cdef double[::1] diff_v = np.ascontiguousarray(diff, dtype=np.float64)
    cdef double[:,::1] si_v = np.ascontiguousarray(Sigma_inv, dtype=np.float64)
    with nogil:
        for i in range(d):
            temp = 0.0
            for j in range(d):
                temp += diff_v[j] * si_v[j, i]
            result += diff_v[i] * temp
    return result


cdef inline double _logistic_c(double cov_value, double a, double cov_50,
                                double q_min, double q_max) nogil:
    cdef double q = q_min + (q_max - q_min) / (1.0 + exp(-a * (cov_value - cov_50)))
    if q < q_min:
        return q_min
    if q > q_max:
        return q_max
    return q


def logistic_mapping(double cov_value, double cov_10, double cov_50, double cov_90,
                     double q_min, double q_max, double alpha):
    cdef double a = alpha / (cov_90 - cov_10 + 1e-12)
    return _logistic_c(cov_value, a, cov_50, q_min, q_max)


def compute_logistic_mapping_vec(cnp.ndarray[double, ndim=1] covs,
                                  double cov_10, double cov_50, double cov_90,
                                  double q_min, double q_max, double alpha):
    cdef int n = covs.shape[0]
    cdef int i
    cdef double a = alpha / (cov_90 - cov_10 + 1e-12)
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef double[::1] covs_v = np.ascontiguousarray(covs, dtype=np.float64)
    with nogil:
        for i in range(n):
            out_v[i] = _logistic_c(covs_v[i], a, cov_50, q_min, q_max)
    return out


def compute_twenn_mu(cnp.ndarray[double, ndim=2] dist):
    cdef int N = dist.shape[0]
    cdef int i, j
    cdef double r1, r2, d
    cdef cnp.ndarray[double, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef double[:,::1] dist_v = np.ascontiguousarray(dist, dtype=np.float64)
    with nogil:
        for i in range(N):
            r1 = DBL_MAX
            r2 = DBL_MAX
            for j in range(N):
                if j == i:
                    continue
                d = dist_v[i, j]
                if d < r1:
                    r2 = r1
                    r1 = d
                elif d < r2:
                    r2 = d
            out_v[i] = (r2 / r1) if r1 > 1e-12 else 1.0
    return out


def compute_persistence_entropy_typed(cnp.ndarray[double, ndim=2] diagram):
    cdef int n = diagram.shape[0]
    cdef int i
    cdef double lt, L_total = 0.0, entropy = 0.0, p
    cdef double[:,::1] dgm_v = np.ascontiguousarray(diagram, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] lifetimes = np.empty(n, dtype=np.float64)
    cdef double[::1] lt_v = lifetimes
    cdef int valid_n = 0

    for i in range(n):
        lt = dgm_v[i, 1] - dgm_v[i, 0]
        if lt > 1e-12:
            lt_v[valid_n] = lt
            L_total += lt
            valid_n += 1

    if valid_n == 0 or L_total <= 1e-12:
        return 0.0

    with nogil:
        for i in range(valid_n):
            p = lt_v[i] / L_total
            entropy -= p * log(p + 1e-12)

    return entropy

# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import gc
import numpy as np
cimport numpy as cnp
from libc.math cimport log

from entrap.constants import PERSISTENCE_ENTROPY_PERCENTILE_FALLBACK
from entrap._core import compute_persistence_entropy_typed

cnp.import_array()


def compute_h0_diagram(cnp.ndarray X, str metric='euclidean'):
    cdef int n = X.shape[0]
    if n < 2:
        return np.array([[0.0, 0.0]], dtype=np.float64)
    try:
        from ripser import ripser as _ripser
        dgms = _ripser(X, distance_matrix=False, maxdim=0, metric=metric)['dgms']
        dgm_h0 = dgms[0]
        finite_mask = ~np.isinf(dgm_h0[:, 1])
        dgm_h0_finite = dgm_h0[finite_mask]
        if len(dgm_h0_finite) == 0:
            return np.array([[0.0, 0.0]], dtype=np.float64)
        return dgm_h0_finite
    except Exception:
        return np.array([[0.0, 0.0]], dtype=np.float64)


def compute_persistence_entropy(cnp.ndarray diagram):
    if len(diagram) == 0:
        return 0.0
    return compute_persistence_entropy_typed(
        np.ascontiguousarray(diagram, dtype=np.float64)
    )


def compute_sequential_persistence_entropy(cnp.ndarray cluster_points,
                                            cnp.ndarray candidates,
                                            cnp.ndarray candidate_indices,
                                            str metric='euclidean'):
    cdef int n_candidates = len(candidates)
    cdef int i

    if n_candidates == 0:
        return np.array([], dtype=np.float64), []

    cdef cnp.ndarray[double, ndim=1] entropy_values = np.zeros(
        n_candidates, dtype=np.float64
    )
    current_data = cluster_points.copy()

    for i in range(n_candidates):
        current_data = np.vstack([current_data, candidates[i].reshape(1, -1)])
        diagram = compute_h0_diagram(current_data, metric=metric)
        entropy_values[i] = compute_persistence_entropy(diagram)
        if (i + 1) % 50 == 0:
            gc.collect()

    return entropy_values, candidate_indices


def detect_knee_with_kneed(cnp.ndarray[double, ndim=1] entropy_values,
                            double fallback_percentile=PERSISTENCE_ENTROPY_PERCENTILE_FALLBACK):
    cdef int n = entropy_values.shape[0]
    cdef int min_idx, auto_accept

    if n == 0:
        return 0
    if n == 1:
        return 1

    min_idx = int(np.argmin(entropy_values))
    auto_accept = min_idx + 1

    if min_idx == n - 1:
        return n

    post_entropy = entropy_values[min_idx:]
    if len(post_entropy) <= 2:
        return auto_accept

    try:
        from kneed import KneeLocator
        kneedle = KneeLocator(
            np.arange(len(post_entropy)),
            post_entropy,
            curve='concave',
            direction='increasing',
            online=True
        )
        if kneedle.knee is not None:
            return max(auto_accept, min(min_idx + int(kneedle.knee), n))
        threshold = np.percentile(post_entropy, fallback_percentile)
        return max(
            auto_accept,
            min(min_idx + int(np.sum(post_entropy <= threshold)), n)
        )
    except Exception:
        return auto_accept

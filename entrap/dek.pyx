# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport floor, fabs

from entrap.constants import DEK_Q_MIN, DEK_Q_MAX, K_MIN, K_MAX, M_MIN, K_PERCENTILE
from entrap.utils import optimize_memory
from entrap._core import compute_cov_from_rows, compute_logistic_mapping_vec
from entrap.intrinsic_dim import estimate_intrinsic_dimension_twenn

cnp.import_array()


def compute_cov_distribution(double[::1] per_point_covs):
    cdef int n = per_point_covs.shape[0]
    cdef double q10, q50, q90, epsilon

    if n == 0:
        return (0.0, 0.5, 1.0)

    arr = np.asarray(per_point_covs)
    finite_array = arr[np.isfinite(arr)]

    if len(finite_array) == 0:
        return (0.0, 0.5, 1.0)

    q10 = float(np.percentile(finite_array, 10))
    q50 = float(np.percentile(finite_array, 50))
    q90 = float(np.percentile(finite_array, 90))

    if fabs(q90 - q10) < 1e-12:
        epsilon = max(1e-6, 0.01 * fabs(q10) if q10 != 0.0 else 1e-6)
        q90 = q10 + epsilon

    return (q10, q50, q90)


cdef class Density_Equalization_K:
    cdef public double q_min, q_max, k_min, k_max, m_min, alpha
    cdef public dict cluster_k_values_
    cdef public dict cluster_intrinsic_dims_
    cdef public dict cluster_basic_stats_
    cdef public bint fitted_
    cdef public double last_estimated_dim_

    def __init__(self, double alpha=10.0):
        self.q_min = DEK_Q_MIN
        self.q_max = DEK_Q_MAX
        self.k_min = K_MIN
        self.k_max = K_MAX
        self.m_min = M_MIN
        self.alpha = alpha
        self.cluster_k_values_ = {}
        self.cluster_intrinsic_dims_ = {}
        self.cluster_basic_stats_ = {}
        self.fitted_ = False
        self.last_estimated_dim_ = 1.0

    cdef int _compute_adaptive_m(self, object X, int n):
        cdef double d_hat, m_raw
        cdef int m_int
        if n <= 2:
            m_int = <int>self.m_min
            return m_int if m_int < n else n - 1
        d_hat = estimate_intrinsic_dimension_twenn(X, X_is_dist=False)
        self.last_estimated_dim_ = d_hat
        m_raw = floor(n ** (1.0 / (d_hat + 1.0)))
        m_int = <int>m_raw
        if m_int < <int>self.m_min:
            m_int = <int>self.m_min
        if m_int > n - 1:
            m_int = n - 1
        return m_int

    @optimize_memory
    def fit(self, object X, object labels):
        from scipy.spatial import cKDTree
        cdef int n_points, k_query, cid_int, i
        cdef double d_hat

        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        unique_clusters = np.unique(labels[labels >= 0])
        self.cluster_k_values_.clear()
        self.cluster_intrinsic_dims_.clear()
        self.cluster_basic_stats_.clear()

        for cid in unique_clusters:
            cid_int = int(cid)
            mask = (labels == cid_int)
            cluster_points = X[mask]
            n_points = cluster_points.shape[0]

            if n_points <= 1:
                self.cluster_k_values_[cid_int] = np.full(n_points, <int>self.k_min, dtype=np.int64)
                self.cluster_intrinsic_dims_[cid_int] = 1.0
                self.cluster_basic_stats_[cid_int] = {
                    'n_points': n_points, 'k_mean': self.k_min, 'k_std': 0.0,
                }
                continue

            m_adaptive = self._compute_adaptive_m(cluster_points, n_points)
            d_hat = self.last_estimated_dim_
            self.cluster_intrinsic_dims_[cid_int] = d_hat

            k_query = min(m_adaptive + 1, n_points)
            tree = cKDTree(cluster_points, leafsize=40, compact_nodes=True, balanced_tree=True)
            distances, _ = tree.query(cluster_points, k=k_query, p=2, workers=-1)

            if k_query > 1:
                neighbor_dist = np.ascontiguousarray(distances[:, 1:], dtype=np.float64)
            else:
                neighbor_dist = np.zeros((n_points, 0), dtype=np.float64)

            if neighbor_dist.shape[1] == 0:
                self.cluster_k_values_[cid_int] = np.full(n_points, <int>self.k_min, dtype=np.int64)
                self.cluster_basic_stats_[cid_int] = {
                    'n_points': n_points, 'k_mean': self.k_min, 'k_std': 0.0,
                }
                continue

            per_point_covs = np.ascontiguousarray(
                compute_cov_from_rows(np.ascontiguousarray(neighbor_dist, dtype=np.float64)),
                dtype=np.float64
            )
            cov_10, cov_50, cov_90 = compute_cov_distribution(per_point_covs)
            q_adaptive = np.ascontiguousarray(
                compute_logistic_mapping_vec(
                    per_point_covs, cov_10, cov_50, cov_90,
                    self.q_min, self.q_max, self.alpha
                ),
                dtype=np.float64
            )

            r_adaptive = np.array([
                float(np.quantile(neighbor_dist[i, :], q_adaptive[i]))
                if neighbor_dist[i, :].size > 0 else 0.0
                for i in range(n_points)
            ], dtype=np.float64)
            r_adaptive = np.maximum(r_adaptive, 1e-12)

            neighbor_lists = tree.query_ball_tree(tree, r=float(r_adaptive.max()), p=2)
            k_values = np.array([
                sum(
                    1 for j in neighbor_lists[i]
                    if i != j and
                    np.linalg.norm(cluster_points[i] - cluster_points[j]) <= r_adaptive[i]
                )
                for i in range(n_points)
            ], dtype=np.int64)

            k_values = np.clip(k_values, <int>self.k_min, <int>self.k_max)
            if np.all(k_values == 0):
                k_values[:] = <int>self.k_min

            self.cluster_k_values_[cid_int] = k_values
            self.cluster_basic_stats_[cid_int] = {
                'n_points': n_points,
                'k_mean': float(np.mean(k_values)),
                'k_std': float(np.std(k_values)),
            }

        self.fitted_ = True
        return self

    def get_k_percentile(self, int cluster_id, double percentile=K_PERCENTILE):
        if not self.fitted_ or cluster_id not in self.cluster_k_values_:
            return <int>self.k_min
        vals = self.cluster_k_values_[cluster_id]
        if vals.size == 0:
            return <int>self.k_min
        return int(np.round(np.percentile(vals, percentile)))

    def get_intrinsic_dimension(self, int cluster_id):
        if not self.fitted_:
            return 1.0
        return self.cluster_intrinsic_dims_.get(cluster_id, 1.0)

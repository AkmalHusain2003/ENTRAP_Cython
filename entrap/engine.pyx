# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import gc
import numpy as np
cimport numpy as cnp
import multiprocessing
from joblib import Parallel, delayed

from entrap.constants import RIDGE_EPSILON, K_MIN
from entrap.utils import validate_metric, optimize_memory
from entrap._core import (
    compute_cluster_mean,
    compute_cluster_covariance,
    compute_mahalanobis_sq,
)
from entrap.tda import (
    compute_h0_diagram,
    compute_persistence_entropy,
    compute_sequential_persistence_entropy,
    detect_knee_with_kneed,
)
from entrap.tracker import EntropyProgressTracker
from entrap.dek import Density_Equalization_K

cnp.import_array()


def _evaluate_cluster_worker(cnp.ndarray X,
                              cnp.ndarray labels,
                              int cid,
                              set candidates_set,
                              double ridge_epsilon,
                              str metric):
    cdef int i, idx, d, cluster_size_initial
    cdef double mahal_sq, dist_val, entropy_initial, entropy_after, entropy_before
    cdef double delta_entropy, log_det_Sigma
    cdef bint accepted

    if len(candidates_set) == 0:
        return {}, None

    candidates_list = [idx for idx in candidates_set if labels[idx] == -1]
    if len(candidates_list) == 0:
        return {}, None

    cluster_mask = labels == cid
    cluster_points = X[cluster_mask]
    cluster_size_initial = len(cluster_points)
    d = cluster_points.shape[1]

    entropy_initial = compute_persistence_entropy(
        compute_h0_diagram(cluster_points, metric=metric)
    )

    mu = compute_cluster_mean(np.ascontiguousarray(cluster_points, dtype=np.float64))
    Sigma_reg = compute_cluster_covariance(
        np.ascontiguousarray(cluster_points, dtype=np.float64),
        np.ascontiguousarray(mu, dtype=np.float64),
        ridge_epsilon,
    )

    try:
        Sigma_inv = np.linalg.inv(Sigma_reg)
        log_det_Sigma = float(np.linalg.slogdet(Sigma_reg)[1])
    except np.linalg.LinAlgError:
        Sigma_inv = np.eye(d) / (np.trace(Sigma_reg) / d + 1e-6)
        log_det_Sigma = 0.0

    candidate_distances = []
    for idx in candidates_list:
        diff = X[idx] - mu
        mahal_sq = compute_mahalanobis_sq(
            np.ascontiguousarray(diff, dtype=np.float64),
            np.ascontiguousarray(Sigma_inv, dtype=np.float64),
        )
        dist_val = (
            0.5 * mahal_sq
            + 0.5 * log_det_Sigma
            + (d / 2.0) * np.log(2.0 * np.pi)
        )
        candidate_distances.append((idx, dist_val))

    candidate_distances.sort(key=lambda item: item[1])

    sorted_indices = np.array([x[0] for x in candidate_distances], dtype=np.int64)
    sorted_mahalanobis = np.array([x[1] for x in candidate_distances], dtype=np.float64)
    sorted_candidates = X[sorted_indices]

    entropy_values, _ = compute_sequential_persistence_entropy(
        cluster_points, sorted_candidates, sorted_indices, metric=metric
    )

    cutoff_index = detect_knee_with_kneed(
        np.ascontiguousarray(entropy_values, dtype=np.float64)
    )

    cluster_eval = {}
    for i in range(len(candidate_distances)):
        idx = candidate_distances[i][0]
        dist_val = candidate_distances[i][1]
        accepted = (i < cutoff_index)
        entropy_after = float(entropy_values[i]) if i < len(entropy_values) else float('inf')
        entropy_before = entropy_initial if i == 0 else float(entropy_values[i - 1])
        delta_entropy = entropy_after - entropy_before
        cluster_eval[idx] = {
            'mahalanobis_distance': dist_val,
            'persistence_entropy': entropy_after,
            'entropy_before': entropy_before,
            'delta_entropy': delta_entropy,
            'accepted': accepted,
            'order': i,
        }

    tracking_data = {
        'cluster_id': cid,
        'candidate_indices': sorted_indices,
        'mahalanobis_distances': sorted_mahalanobis,
        'entropy_values': np.asarray(entropy_values, dtype=np.float64),
        'knee_index': cutoff_index,
        'cluster_size_initial': cluster_size_initial,
        'cluster_size_final': cluster_size_initial,
    }

    gc.collect()
    return cluster_eval, tracking_data


cdef class Geometric_Persistence_Entropy_Engine:
    cdef public double ridge_epsilon
    cdef public object metric
    cdef public dict metric_params
    cdef public bint use_memmap
    cdef public bint enable_tracking
    cdef public object tracker

    def __init__(self,
                 double ridge_epsilon=RIDGE_EPSILON,
                 metric='euclidean',
                 bint use_memmap=True,
                 bint enable_tracking=False,
                 **metric_params):
        self.ridge_epsilon = ridge_epsilon
        self.metric = validate_metric(metric)
        self.metric_params = metric_params
        self.use_memmap = use_memmap
        self.enable_tracking = enable_tracking
        self.tracker = EntropyProgressTracker() if enable_tracking else None

    cdef dict _identify_candidates(self,
                                    cnp.ndarray X,
                                    cnp.ndarray labels,
                                    list cluster_ids,
                                    object dek_selector):
        from scipy.spatial import cKDTree
        cdef int cid_int, k_adaptive, k_query

        cluster_candidate_sets = {}
        noise_indices = np.where(labels == -1)[0]

        if len(noise_indices) == 0:
            return {cid: set() for cid in cluster_ids}

        noise_points = X[noise_indices]
        noise_tree = cKDTree(noise_points, compact_nodes=True, balanced_tree=True)

        for cid in cluster_ids:
            cid_int = int(cid)
            cluster_mask = labels == cid_int
            cluster_points = X[cluster_mask]

            if dek_selector is not None:
                k_adaptive = dek_selector.get_k_percentile(cid_int)
            else:
                k_adaptive = <int>K_MIN

            k_query = min(k_adaptive, len(cluster_points))
            distances, indices = noise_tree.query(cluster_points, k=k_query, workers=-1)

            if cluster_points.shape[0] == 1:
                distances = distances.reshape(1, -1)
                indices = indices.reshape(1, -1)

            candidate_local = np.unique(indices.ravel())
            candidate_local = candidate_local[candidate_local < len(noise_points)]

            if len(candidate_local) > 0:
                cluster_candidate_sets[cid_int] = set(
                    int(x) for x in noise_indices[candidate_local]
                )
            else:
                cluster_candidate_sets[cid_int] = set()

        return cluster_candidate_sets

    cdef tuple _resolve_conflicts(self,
                                   cnp.ndarray labels,
                                   dict cluster_evaluations):
        cdef int candidate_idx, winner_cid, total_rescued = 0

        refined_labels = labels.copy()
        all_accepted = set()
        for cid, evals in cluster_evaluations.items():
            for idx, info in evals.items():
                if info['accepted']:
                    all_accepted.add(idx)

        for candidate_idx in all_accepted:
            if refined_labels[candidate_idx] != -1:
                continue
            competing = []
            for cid, evals in cluster_evaluations.items():
                if candidate_idx in evals and evals[candidate_idx]['accepted']:
                    competing.append((cid, evals[candidate_idx]['delta_entropy']))
            if not competing:
                continue
            winner_cid = min(competing, key=lambda x: x[1])[0]
            refined_labels[candidate_idx] = winner_cid
            total_rescued += 1

        return refined_labels, total_rescued

    cdef dict _compute_final_stats(self,
                                    cnp.ndarray labels,
                                    list cluster_ids,
                                    dict cluster_evaluations):
        cdef int cid_int, rescued
        cluster_stats = {}
        for cid in cluster_ids:
            cid_int = int(cid)
            evals = cluster_evaluations.get(cid_int, {})
            rescued = sum(
                1 for idx, info in evals.items()
                if info['accepted'] and labels[idx] == cid_int
            )
            cluster_stats[cid_int] = {
                'rescued': rescued,
                'candidates_evaluated': len(evals),
                'final_size': int(np.sum(labels == cid_int)),
            }
        return cluster_stats

    def reassign_parallel(self,
                          cnp.ndarray X,
                          cnp.ndarray labels,
                          dek_selector=None,
                          int n_jobs=-1):
        cdef int cid_int

        refined_labels = labels.copy()
        if not (refined_labels == -1).any():
            return refined_labels, 0, {}

        unique_labels = np.unique(refined_labels[refined_labels >= 0])
        if len(unique_labels) == 0:
            return refined_labels, 0, {}

        cluster_sizes = [
            (int(cid), int(np.sum(refined_labels == cid))) for cid in unique_labels
        ]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_cluster_ids = [x[0] for x in cluster_sizes]

        cluster_candidate_sets = self._identify_candidates(
            X, refined_labels, sorted_cluster_ids, dek_selector
        )

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        results_list = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_evaluate_cluster_worker)(
                X, refined_labels, cid,
                cluster_candidate_sets[cid],
                self.ridge_epsilon, self.metric,
            )
            for cid in sorted_cluster_ids
        )

        cluster_evaluations = {}
        tracking_data_list = []
        for cid, (cluster_eval, tracking_data) in zip(sorted_cluster_ids, results_list):
            cluster_evaluations[cid] = cluster_eval
            if tracking_data is not None:
                tracking_data_list.append((cid, tracking_data))

        refined_labels, total_rescued = self._resolve_conflicts(
            refined_labels, cluster_evaluations
        )

        if self.enable_tracking and self.tracker is not None:
            for cid, td in tracking_data_list:
                td['cluster_size_final'] = int(np.sum(refined_labels == cid))
                self.tracker.record_cluster_progress(**td)

        cluster_stats = self._compute_final_stats(
            refined_labels, sorted_cluster_ids, cluster_evaluations
        )

        return refined_labels, total_rescued, cluster_stats

    @optimize_memory
    def reassign(self, cnp.ndarray X, cnp.ndarray labels, dek_selector=None):
        return self.reassign_parallel(X, labels, dek_selector, n_jobs=1)

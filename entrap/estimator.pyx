# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import gc
import time
import warnings
import numpy as np
cimport numpy as cnp

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array

from entrap.constants import RIDGE_EPSILON
from entrap.utils import validate_metric
from entrap.engine import Geometric_Persistence_Entropy_Engine
from entrap.dek import Density_Equalization_K
from entrap.results import ENTRAP_Results

warnings.filterwarnings('ignore')

cnp.import_array()


class ENTRAP(BaseEstimator, ClusterMixin):

    def __init__(self,
                 min_cluster_size=30,
                 min_samples=None,
                 ridge_epsilon=RIDGE_EPSILON,
                 metric='euclidean',
                 metric_params=None,
                 use_memmap=True,
                 enable_tracking=False,
                 n_jobs=-1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.ridge_epsilon = ridge_epsilon
        self.metric = validate_metric(metric)
        self.metric_params = metric_params or {}
        self.use_memmap = use_memmap
        self.enable_tracking = enable_tracking
        self.n_jobs = n_jobs
        self.engine = Geometric_Persistence_Entropy_Engine(
            ridge_epsilon=ridge_epsilon,
            metric=metric,
            use_memmap=use_memmap,
            enable_tracking=enable_tracking,
            **self.metric_params,
        )
        self.labels_ = None
        self.result_ = None

    def fit(self, X, y=None):
        from hdbscan import HDBSCAN
        cdef double start_time
        cdef int n_clusters_final

        start_time = time.time()
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
        hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples or self.min_cluster_size,
            metric=self.metric,
            algorithm='best',
            core_dist_n_jobs=-1,
            **self.metric_params,
        )
        hdbscan.fit(X)
        labels = hdbscan.labels_

        dek_selector = Density_Equalization_K()
        try:
            dek_selector.fit(X, labels)
        except Exception:
            dek_selector = None

        try:
            labels_refined, n_rescued, cluster_stats = self.engine.reassign_parallel(
                X, labels, dek_selector, n_jobs=self.n_jobs
            )
        except Exception as e:
            raise RuntimeError(f"Refinement failed: {e}")

        self.labels_ = labels_refined
        elapsed = time.time() - start_time
        n_clusters_final = len(np.unique(labels_refined[labels_refined >= 0]))

        self.result_ = ENTRAP_Results(
            labels=labels_refined,
            probabilities=hdbscan.probabilities_,
            noise_rescued=n_rescued,
            execution_time=elapsed,
            n_clusters=n_clusters_final,
            cluster_stats=cluster_stats,
            hyperparameters={
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples or self.min_cluster_size,
                'ridge_epsilon': self.ridge_epsilon,
                'metric': self.metric,
                'n_jobs': self.n_jobs,
            },
            tracker=self.engine.tracker if self.enable_tracking else None,
        )
        gc.collect()
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def get_summary(self):
        cdef int total_evals, total_rescued
        if self.labels_ is None:
            raise ValueError("Must call fit() first")
        unique_labels = np.unique(self.labels_[self.labels_ >= 0])
        cluster_sizes = [int(np.sum(self.labels_ == l)) for l in unique_labels]
        total_evals = sum(s['candidates_evaluated'] for s in self.result_.cluster_stats.values())
        total_rescued = sum(s['rescued'] for s in self.result_.cluster_stats.values())
        return {
            'n_clusters': self.result_.n_clusters,
            'n_noise': int(np.sum(self.labels_ == -1)),
            'cluster_sizes': cluster_sizes,
            'noise_rescued': self.result_.noise_rescued,
            'time_seconds': round(self.result_.execution_time, 3),
            'total_candidates_evaluated': total_evals,
            'evaluations_per_rescued': round(total_evals / max(total_rescued, 1), 2),
            'n_jobs_used': self.n_jobs if self.n_jobs > 0 else 'auto',
        }

    def _check_tracking(self):
        if not self.enable_tracking or self.result_ is None or self.result_.tracker is None:
            raise ValueError("Tracking tidak diaktifkan. Set enable_tracking=True.")

    def get_rejected_candidates(self, cluster_id):
        self._check_tracking()
        return self.result_.tracker.get_rejected_details(cluster_id)

    def get_accepted_candidates(self, cluster_id):
        self._check_tracking()
        return self.result_.tracker.get_accepted_details(cluster_id)

    def plot_rejected_analysis(self, cluster_id, save_path=None):
        self._check_tracking()
        self.result_.tracker.plot_rejected_analysis(cluster_id, save_path=save_path)

    def plot_entropy_curve(self, cluster_id, save_path=None):
        self._check_tracking()
        self.result_.tracker.plot_entropy_curve(cluster_id, save_path=save_path)

    def plot_comparison(self, cluster_ids, save_path=None):
        self._check_tracking()
        self.result_.tracker.plot_comparison(cluster_ids, save_path=save_path)

    def export_entropy_summary(self):
        self._check_tracking()
        return self.result_.tracker.export_summary()

    def list_tracked_clusters(self):
        self._check_tracking()
        return self.result_.tracker.list_clusters()

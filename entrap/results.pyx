# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef class ENTRAP_Results:
    cdef public cnp.ndarray labels
    cdef public cnp.ndarray probabilities
    cdef public int noise_rescued
    cdef public double execution_time
    cdef public int n_clusters
    cdef public dict cluster_stats
    cdef public object hyperparameters
    cdef public object tracker

    def __init__(self,
                 cnp.ndarray labels,
                 cnp.ndarray probabilities,
                 int noise_rescued,
                 double execution_time,
                 int n_clusters,
                 dict cluster_stats,
                 hyperparameters=None,
                 tracker=None):
        self.labels = labels
        self.probabilities = probabilities
        self.noise_rescued = noise_rescued
        self.execution_time = execution_time
        self.n_clusters = n_clusters
        self.cluster_stats = cluster_stats
        self.hyperparameters = hyperparameters
        self.tracker = tracker

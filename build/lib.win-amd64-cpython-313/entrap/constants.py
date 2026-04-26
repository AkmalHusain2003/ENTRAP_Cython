M_MIN = 2.0
K_MIN, K_MAX = 10.0, 50.0
DEK_Q_MIN, DEK_Q_MAX = 0.1, 0.9
K_PERCENTILE = 50.0
RIDGE_EPSILON = 1e-6
PERSISTENCE_ENTROPY_PERCENTILE_FALLBACK = 75.0

SUPPORTED_METRICS = {
    'euclidean', 'manhattan', 'cityblock', 'minkowski', 'chebyshev',
    'cosine', 'correlation', 'hamming', 'jaccard', 'canberra',
    'braycurtis', 'mahalanobis', 'seuclidean', 'sqeuclidean'
}

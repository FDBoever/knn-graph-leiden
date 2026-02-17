import numpy as np
import hnswlib
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------
# Build approximate kNN graph using HNSW
# ---------------------------------------------------------
def build_knn_hnsw(
    data,
    k=30,
    metric="cosine",
    ef=200,
    M=16,
    ef_query=100,
    random_seed=42
):
    """
    Build approximate kNN using HNSWlib.

    Returns
    -------
    labels : np.ndarray
        Indices of k nearest neighbors (n_samples x k)
    distances : np.ndarray
        Corresponding distances
    """
    n_samples, dim = data.shape
    space = "cosine" if metric == "cosine" else "l2"

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=n_samples, ef_construction=ef, M=M, random_seed=random_seed)
    p.add_items(data)
    p.set_ef(ef_query)

    labels, distances = p.knn_query(data, k=k)
    #return labels, distances
    return labels.astype(np.int32), distances.astype(np.float32)


# ---------------------------------------------------------
# Build exact kNN using scikit-learn
# ---------------------------------------------------------
def build_knn_exact(data, k=30, metric="cosine", n_jobs=1):
    """
    Build exact kNN using scikit-learn.

    Returns
    -------
    labels : np.ndarray
        Indices of k nearest neighbors (n_samples x k)
    distances : np.ndarray
        Corresponding distances
    """
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=n_jobs)
    nn.fit(data)
    distances, labels = nn.kneighbors(data)
    #return labels, distances
    return labels.astype(np.int32), distances.astype(np.float32)

# ---------------------------------------------------------
# Unified kNN interface
# ---------------------------------------------------------
def build_knn(
    data,
    k=30,
    metric="cosine",
    method="auto",
    n_jobs=1,
    hnsw_kwargs=None
):
    """
    Build kNN graph, choosing approximate or exact method automatically.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples x n_features)
    k : int
        Number of nearest neighbors
    metric : str
        "cosine" or "euclidean"
    method : str
        "auto" | "exact" | "hnsw"
        - "auto": use HNSW if n_samples > 10000
    n_jobs : int
        Threads for exact kNN
    hnsw_kwargs : dict
        Additional HNSW parameters: ef, M, ef_query, random_seed

    Returns
    -------
    labels : np.ndarray
        kNN indices
    distances : np.ndarray
        kNN distances
    """
    n_samples = data.shape[0]

    if hnsw_kwargs is None:
        hnsw_kwargs = {}

    if method == "exact" or (method == "auto" and n_samples <= 10000):
        labels, distances = build_knn_exact(data, k=k, metric=metric, n_jobs=n_jobs)
    else:
        labels, distances = build_knn_hnsw(data, k=k, metric=metric, **hnsw_kwargs)

    #return labels, distances
    return labels.astype(np.int32), distances.astype(np.float32)

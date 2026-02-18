import numpy as np
import hnswlib
from sklearn.neighbors import NearestNeighbors

#approximate kin
def build_knn_hnsw(
    data,
    k=30,
    metric="cosine",
    ef=200,
    M=16,
    ef_query=100,
    random_seed=42
):
    n_samples, dim = data.shape
    space = "cosine" if metric == "cosine" else "l2"

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=n_samples, ef_construction=ef, M=M, random_seed=random_seed)
    p.add_items(data)
    p.set_ef(ef_query)

    labels, distances = p.knn_query(data, k=k)
    #return labels, distances
    return labels.astype(np.int32), distances.astype(np.float32)



# exact kin using scikit-learn
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


def build_knn(
    data,
    k=30,
    metric="cosine",
    method="auto",
    n_jobs=1,
    hnsw_kwargs=None
):
    n_samples = data.shape[0]

    if hnsw_kwargs is None:
        hnsw_kwargs = {}

    if method == "exact" or (method == "auto" and n_samples <= 10000):
        labels, distances = build_knn_exact(data, k=k, metric=metric, n_jobs=n_jobs)
    else:
        labels, distances = build_knn_hnsw(data, k=k, metric=metric, **hnsw_kwargs)

    #return labels, distances
    return labels.astype(np.int32), distances.astype(np.float32)

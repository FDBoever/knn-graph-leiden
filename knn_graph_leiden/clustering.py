import numpy as np
import igraph as ig
import leidenalg
from tqdm import tqdm


# ---------------------------------------------------------
# Run Leiden clustering on igraph graph
# ---------------------------------------------------------
def run_leiden(A, resolution=1.0, n_iterations=-1, seed=42):
    """
    Run Leiden community detection on a weighted graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Weighted adjacency matrix (graph)
    resolution : float
        Leiden resolution parameter
    n_iterations : int
        Maximum number of iterations (-1 for default behaviour)
    seed : int
        Random seed for Leiden reproducibility

    Returns
    -------
    labels : np.ndarray
        Cluster membership for each node
    modularity : float
        Graph modularity for clustering
    """

    if A.nnz == 0:
        print("[Warning] Graph has no edges. Cannot run Leiden.")
        return np.array([]), np.nan

    # Convert sparse matrix to igraph
    sources, targets = A.nonzero()
    weights = A.data

    g = ig.Graph(directed=False)
    g.add_vertices(A.shape[0])
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights

    try:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed
        )

        labels = np.array(partition.membership)
        modularity = g.modularity(labels, weights=g.es["weight"])

        return labels, modularity

    except Exception as e:
        print(f"[Error] Leiden clustering failed: {e}")
        return np.array([]), np.nan


# ---------------------------------------------------------
# Resolution sweep for cluster stability
# ---------------------------------------------------------
def leiden_resolution_sweep(A, resolutions=None, n_iterations=-1, seed=42):
    """
    Run Leiden clustering over a sweep of resolution parameters.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Weighted adjacency matrix
    resolutions : list or np.ndarray
        List of resolution parameters to try
    n_iterations : int
        Maximum iterations for Leiden
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with resolution as key and tuple (labels, modularity)
    """

    if resolutions is None:
        resolutions = np.linspace(0.01, 1.0, 10)

    results = dict()

    for r in tqdm(resolutions, desc="Resolution sweep"):
        labels, modularity = run_leiden(
            A,
            resolution=r,
            n_iterations=n_iterations,
            seed=seed
        )
        results[r] = (labels, modularity)

    return results


# ---------------------------------------------------------
# Merge tiny clusters (optional post-processing)
# ---------------------------------------------------------
def merge_micro_clusters(labels, min_size=5, seed=42):
    """
    Merge clusters smaller than min_size into nearest larger cluster.

    NOTE:
    Current implementation assigns randomly to a large cluster.
    For production use, you may want centroid-based reassignment.

    Parameters
    ----------
    labels : np.ndarray
        Cluster assignments
    min_size : int
        Minimum allowed cluster size
    seed : int
        Random seed for deterministic reassignment

    Returns
    -------
    np.ndarray
        Updated cluster labels
    """

    rng = np.random.default_rng(seed)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)

    small_clusters = unique[counts < min_size]
    large_clusters = unique[counts >= min_size]

    for sc in small_clusters:
        idx = np.where(labels == sc)[0]

        if len(large_clusters) > 0:
            labels[idx] = rng.choice(large_clusters)
        else:
            labels[idx] = 0  # fallback

    return labels
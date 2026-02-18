import numpy as np
import igraph as ig
import leidenalg
from tqdm import tqdm


# Run Leiden clustering on igraph graph
def run_leiden(A, resolution=1.0, n_iterations=-1, seed=42):

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


# resolution sweep for cluster stability
def leiden_resolution_sweep(A, resolutions=None, n_iterations=-1, seed=42):

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


# merge tiny clusters (optional post-processing)
def merge_micro_clusters(labels, min_size=5, seed=42):

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
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import igraph as ig

# ---------------------------------------------------------
# Graph-based modularity
# ---------------------------------------------------------
def compute_modularity(graph, labels):
    """
    Compute modularity of a clustering on an igraph graph.

    Parameters
    ----------
    graph : igraph.Graph
        Graph object with weights
    labels : array-like
        Cluster assignments for each node

    Returns
    -------
    float
        Modularity score
    """
    membership = np.array(labels)
    if "weight" in graph.edge_attributes():
        weights = graph.es["weight"]
        return graph.modularity(membership, weights=weights)
    else:
        return graph.modularity(membership)


# ---------------------------------------------------------
# Silhouette score (feature space)
# ---------------------------------------------------------
def compute_silhouette(X, labels, metric="cosine"):
    """
    Compute silhouette score in high-dimensional feature space.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples x n_features)
    labels : array-like
        Cluster assignments
    metric : str
        Distance metric for silhouette

    Returns
    -------
    float
        Silhouette score
    """
    try:
        if len(np.unique(labels)) <= 1:
            return np.nan
        return silhouette_score(X, labels, metric=metric)
    except Exception as e:
        print(f"[Warning] Silhouette score failed: {e}")
        return np.nan


# ---------------------------------------------------------
# Davies-Bouldin score
# ---------------------------------------------------------
def compute_davies_bouldin(X, labels):
    """
    Compute Davies-Bouldin score in high-dimensional feature space.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : array-like
        Cluster assignments

    Returns
    -------
    float
        Davies-Bouldin score
    """
    try:
        if len(np.unique(labels)) <= 1:
            return np.nan
        return davies_bouldin_score(X, labels)
    except Exception as e:
        print(f"[Warning] Davies-Bouldin score failed: {e}")
        return np.nan


# ---------------------------------------------------------
# Aggregate all metrics
# ---------------------------------------------------------
def evaluate_clustering(X, graph, labels):
    """
    Compute a dictionary of clustering metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    graph : igraph.Graph
        Weighted graph used for clustering
    labels : array-like
        Cluster assignments

    Returns
    -------
    dict
        Metrics: n_clusters, modularity, silhouette, davies_bouldin
    """
    labels = np.array(labels)
    metrics = dict()
    metrics["n_clusters"] = len(np.unique(labels))
    metrics["modularity"] = None
    if graph is not None:
        metrics["modularity"] = graph.modularity(labels, weights=graph.es["weight"])
    metrics["silhouette"] = compute_silhouette(X, labels)
    metrics["davies_bouldin"] = compute_davies_bouldin(X, labels)
    return metrics
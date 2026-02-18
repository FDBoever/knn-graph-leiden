import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import igraph as ig

def compute_modularity(graph, labels):
    membership = np.array(labels)
    if "weight" in graph.edge_attributes():
        weights = graph.es["weight"]
        return graph.modularity(membership, weights=weights)
    else:
        return graph.modularity(membership)

def compute_silhouette(X, labels, metric="cosine"):
    try:
        if len(np.unique(labels)) <= 1:
            return np.nan
        return silhouette_score(X, labels, metric=metric)
    except Exception as e:
        print(f"[Warning] Silhouette score failed: {e}")
        return np.nan


def compute_davies_bouldin(X, labels):
    try:
        if len(np.unique(labels)) <= 1:
            return np.nan
        return davies_bouldin_score(X, labels)
    except Exception as e:
        print(f"[Warning] Davies-Bouldin score failed: {e}")
        return np.nan


def evaluate_clustering(X, graph, labels):
    labels = np.array(labels)
    metrics = dict()
    metrics["n_clusters"] = len(np.unique(labels))
    metrics["modularity"] = None
    if graph is not None:
        metrics["modularity"] = graph.modularity(labels, weights=graph.es["weight"])
    metrics["silhouette"] = compute_silhouette(X, labels)
    metrics["davies_bouldin"] = compute_davies_bouldin(X, labels)
    return metrics
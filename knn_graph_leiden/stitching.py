import numpy as np
from scipy.sparse import csr_matrix

def compute_cluster_connectivity(A: csr_matrix, labels):
    """
    Compute inter- and intra-cluster edge weights.
    Returns:
        intra_weights: dict cluster -> weight
        inter_weights: dict (c1,c2) -> weight
    """
    labels = np.array(labels)
    clusters = np.unique(labels)
    cluster_map = {c: i for i, c in enumerate(clusters)}
    n_clusters = len(clusters)

    intra = np.zeros(n_clusters)
    inter = {}

    A = A.tocoo()

    for i, j, w in zip(A.row, A.col, A.data):
        ci = cluster_map[labels[i]]
        cj = cluster_map[labels[j]]

        if ci == cj:
            intra[ci] += w
        else:
            key = tuple(sorted((ci, cj)))
            inter[key] = inter.get(key, 0) + w

    return clusters, intra, inter

def find_merges(clusters, intra, inter, threshold=0.3):
    """
    Identify cluster pairs to merge based on connectivity ratio.
    """
    merges = []

    for (c1, c2), w_inter in inter.items():
        ratio = w_inter / min(intra[c1], intra[c2] + 1e-12)
        if ratio > threshold:
            merges.append((c1, c2))

    return merges


def merge_clusters(labels, merges):
    """
    Apply merges to label array using union-find.
    """
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b in merges:
        union(a, b)

    new_labels = []
    for l in labels:
        new_labels.append(find(l))

    # relabel consecutively
    unique = {c: i for i, c in enumerate(np.unique(new_labels))}
    new_labels = np.array([unique[c] for c in new_labels])

    return new_labels

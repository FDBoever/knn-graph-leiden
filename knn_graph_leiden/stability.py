# graph_pipeline/stability.py

import numpy as np

def detect_main_clusters(labels, top_fraction=0.99):
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    # Sort descending by size
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    total_points = sum(counts)
    cumulative = 0
    main_clusters = []
    for cl, count in sorted_clusters:
        main_clusters.append(cl)
        cumulative += count
        if cumulative / total_points >= top_fraction:
            break
    return main_clusters


def merge_small_clusters(X, labels, main_clusters):
    labels = np.array(labels)
    new_labels = labels.copy()
    original_labels = labels.copy()

    # Compute centroids of main clusters
    main_centroids = {}
    for cl in main_clusters:
        main_centroids[cl] = X[labels == cl].mean(axis=0)

    # Merge each tiny cluster
    tiny_clusters = [cl for cl in np.unique(labels) if cl not in main_clusters]
    for cl in tiny_clusters:
        points_idx = np.where(labels == cl)[0]
        cluster_centroid = X[points_idx].mean(axis=0)
        # Find nearest main cluster
        distances = {mcl: np.linalg.norm(cluster_centroid - main_centroids[mcl]) for mcl in main_clusters}
        nearest_main = min(distances, key=distances.get)
        # Reassign points
        new_labels[points_idx] = nearest_main

    return new_labels, original_labels

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples

sns.set(style="white")  # clean style globally

# ---------------------------------------------------------
# Helper: generate color palette based on number of clusters
# ---------------------------------------------------------
def get_palette(n_clusters):
    """
    Return visually distinct colors for n_clusters.
    - 2 clusters: blue/orange divergent
    - <=10: tab10
    - <=20: tab20
    - >20: hls
    """
    if n_clusters == 2:
        return ["#1f77b4", "#ff7f0e"]  # blue/orange
    elif n_clusters <= 10:
        return sns.color_palette("tab10", n_clusters)
    elif n_clusters <= 20:
        return sns.color_palette("tab20", n_clusters)
    else:
        return sns.color_palette("hls", n_clusters)

# ---------------------------------------------------------
# UMAP / 2D embedding
# ---------------------------------------------------------
def plot_embedding(
    embedding,
    labels,
    output_dir=".",
    save_name="embedding.png",
    title="2D projection colored by clusters",
    legend=True,
    point_size=4
):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    palette = get_palette(len(unique_labels))

    plt.figure(figsize=(8, 6))
    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=palette[i],
            s=point_size,
            alpha=0.7,
            label=f"Cluster {cl}",
            linewidths=0
        )

    ax = plt.gca()
    ax.set_xlabel("Dim1", fontsize=10)
    ax.set_ylabel("Dim2", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)

    if legend:
        plt.legend(markerscale=1.2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

# ---------------------------------------------------------
# Cluster size bar plot
# ---------------------------------------------------------
def plot_cluster_sizes(labels, output_dir=".", save_name="cluster_sizes.png"):
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    clusters = counts.index.astype(str)
    values = counts.values
    palette = get_palette(len(clusters))

    plt.figure(figsize=(8, 6))
    plt.barh(clusters, values, color=palette)
    plt.xlabel("Number of samples")
    plt.ylabel("Cluster")
    plt.title("Cluster sizes")
    plt.gca().invert_yaxis()  # largest at top
    plt.grid(False)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

# ---------------------------------------------------------
# Silhouette scores per cluster
# ---------------------------------------------------------
def plot_silhouette(embedding, labels, output_dir=".", save_name="silhouette.png"):
    labels = np.array(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        print("[Warning] Less than 2 clusters: silhouette cannot be computed.")
        return

    sil_vals = silhouette_samples(embedding, labels)
    sil_df = pd.DataFrame({"cluster": labels.astype(str), "silhouette": sil_vals})
    medians = sil_df.groupby("cluster")["silhouette"].median().sort_values(ascending=False).index
    palette = get_palette(n_clusters)

    plt.figure(figsize=(8, 6))
    for i, cl in enumerate(medians):
        cluster_data = sil_df[sil_df["cluster"] == cl]["silhouette"]
        parts = plt.violinplot(cluster_data, positions=[i], showmeans=True, showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(palette[int(cl) % n_clusters])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmeans'].set_color('red')

    plt.xticks(range(len(medians)), medians)
    plt.xlabel("Silhouette score")
    plt.ylabel("Cluster")
    plt.title("Silhouette scores per cluster")
    plt.grid(False)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

# ---------------------------------------------------------
# Optional: inter-cluster similarity heatmap
# ---------------------------------------------------------
def plot_cluster_heatmap(adj_matrix, labels, output_dir=".", save_name="cluster_heatmap.png"):
    labels = np.array(labels)
    clusters = np.unique(labels)
    cluster_matrix = np.zeros((len(clusters), len(clusters)))

    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            mask_i = labels == ci
            mask_j = labels == cj
            if np.any(mask_i) and np.any(mask_j):
                cluster_matrix[i, j] = adj_matrix[np.ix_(mask_i, mask_j)].mean()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cluster_matrix, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=clusters, yticklabels=clusters, cbar_kws={"label": "Mean weight"}
    )
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.title("Inter-cluster similarity")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

def plot_embedding_grid(embedding, clusters_dict, output_dir=".", save_name="embedding_sweep.png"):
    import math

    n_plots = len(clusters_dict)
    if n_plots == 0:
        return

    cols = 4
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n_plots:]:
        ax.axis("off")

    for idx, (res, (labels, _)) in enumerate(sorted(clusters_dict.items())):
        ax = axes[idx]
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        palette = get_palette(len(unique_labels))

        for i, cl in enumerate(unique_labels):
            mask = labels == cl
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=3,
                alpha=0.7,
                color=palette[i],
                linewidths=0
            )

        ax.set_title(f"r={res:.3f} | k={len(unique_labels)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

# ---------------------------------------------------------
# Summary plot generator
# ---------------------------------------------------------
def generate_summary_plots(embedding, labels, adjacency_matrix=None, output_dir=".", prefix="plot"):
    plot_embedding(embedding, labels, output_dir, f"{prefix}_embedding.png", "UMAP Projection")
    plot_cluster_sizes(labels, output_dir, f"{prefix}_cluster_sizes.png")
    plot_silhouette(embedding, labels, output_dir, f"{prefix}_silhouette.png")
    if adjacency_matrix is not None:
        plot_cluster_heatmap(adjacency_matrix, labels, output_dir, f"{prefix}_cluster_heatmap.png")

# ---------------------------------------------------------
#  plot metrics vs resolution (sweep)
# ---------------------------------------------------------
def plot_resolution_metrics(metrics_df, output_dir=".", prefix="results"):
    """
    Plot clustering metrics as a function of resolution.
    Generates a composite multi-panel figure.
    """

    if metrics_df.empty or "resolution" not in metrics_df.columns:
        print("   - No resolution metrics available for plotting.")
        return

    # Sort by resolution
    metrics_df = metrics_df.sort_values("resolution")

    # Identify metric columns (exclude resolution)
    metric_cols = [c for c in metrics_df.columns if c != "resolution"]

    if len(metric_cols) == 0:
        print("   - No metric columns found.")
        return

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(6, 3 * n_metrics), sharex=True)

    if n_metrics == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        ax.plot(metrics_df["resolution"], metrics_df[col], marker="o")
        ax.set_ylabel(col)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[-1].set_xlabel("Resolution")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{prefix}_resolution_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"   - resolution metric plot saved: {save_path}")
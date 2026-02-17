import numpy as np


def recommend_resolution(metrics_df, tolerance=0.02):
    """
    Recommend an optimal Leiden resolution based on composite scoring.
    Returns recommended resolution and scored dataframe.
    """

    df = metrics_df.copy()

    # Normalize metrics to [0,1]
    def minmax(x):
        if x.max() == x.min():
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    df["sil_scaled"] = minmax(df["silhouette"])
    df["mod_scaled"] = minmax(df["modularity"])
    df["db_scaled"] = minmax(df["davies_bouldin"])
    df["k_scaled"] = minmax(df["n_clusters"])

    # Composite score
    df["composite_score"] = (
        0.4 * df["sil_scaled"] +
        0.3 * df["mod_scaled"] +
        0.2 * (1 - df["db_scaled"]) +
        0.1 * (1 - df["k_scaled"])
    )

    best_score = df["composite_score"].max()

    # Keep resolutions within tolerance of best score
    candidates = df[df["composite_score"] >= best_score - tolerance]

    # Prefer lowest resolution among near-optimal candidates
    recommended_row = candidates.sort_values("resolution").iloc[0]

    return recommended_row["resolution"], df
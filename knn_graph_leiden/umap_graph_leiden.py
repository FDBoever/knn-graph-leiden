#!/usr/bin/env python

import argparse
import os
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

from knn_graph_leiden.graph import to_igraph
from knn_graph_leiden.clustering import run_leiden, leiden_resolution_sweep
from knn_graph_leiden.visualisation import (
    plot_embedding,
    plot_embedding_grid
)


# ---------------------------------------------------------
# argparse
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="UMAP-topology-based clustering using Leiden or DBSCAN with optional sweeps"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--prefix", type=str, default=None)

    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_dim", type=int, default=20)

    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--l2norm", action="store_true")

    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--resolution_sweep", action="store_true")  # Leiden sweep

    parser.add_argument("--dbscan", action="store_true")
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=10)
    parser.add_argument("--dbscan_sweep", action="store_true")  # DBSCAN sweep

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.index.values, df.values.astype(np.float32)


# ---------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    def out_path(fname):
        return os.path.join(args.output, f"{prefix}_{fname}")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    ids, X = load_data(args.input)
    if args.l2norm:
        X = normalize(X)

    # -----------------------------------------------------
    # UMAP topology inference (20D)
    # -----------------------------------------------------
    print("Running UMAP (topology inference)...")
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_dim,
        metric=args.metric,
        random_state=args.seed
    )
    embedding_20d = reducer.fit_transform(X)
    graph = reducer.graph_
    print(f"UMAP graph: {graph.nnz} edges")

    # -----------------------------------------------------
    # Leiden clustering
    # -----------------------------------------------------
    print("Running Leiden clustering...")
    clusters_dict = {}
    if args.resolution_sweep:
        resolutions = [
            0.001, 0.0025, 0.005, 0.0075,
            0.01, 0.025, 0.05, 0.075,
            0.1, 0.25, 0.5, 0.75,
            1.0, 2.5, 5.0
        ]
        clusters_dict = leiden_resolution_sweep(graph, resolutions=resolutions, seed=args.seed)

        sweep_df = pd.DataFrame({"id": ids})
        for r in sorted(clusters_dict.keys()):
            labels, _ = clusters_dict[r]
            sweep_df[f"r_{r:.4f}"] = labels
        sweep_df.to_csv(out_path("clusters_sweep.tsv"), sep="\t", index=False)
        print("Leiden sweep clusters saved")

        last_res = sorted(clusters_dict.keys())[-1]
        last_labels = clusters_dict[last_res][0]
    else:
        labels, modularity = run_leiden(graph, resolution=args.resolution, seed=args.seed)
        clusters_dict[args.resolution] = (labels, modularity)
        last_labels = labels
        pd.DataFrame({"id": ids, "cluster": labels}).to_csv(out_path("leiden_clusters.tsv"), sep="\t", index=False)
        print("Leiden clustering saved")

    # -----------------------------------------------------
    # DBSCAN clustering
    # -----------------------------------------------------
    db_labels = None
    db_clusters_dict = {}
    if args.dbscan:
        if args.dbscan_sweep:
            # fixed eps grid
            eps_list = [0.01, 0.025, 0.05, 0.075,
                        0.1, 0.25, 0.5, 0.75,
                        1.0, 2.5, 5.0]
            
            print(f"Running DBSCAN sweep over eps: {eps_list}")

            sweep_df = pd.DataFrame({"id": ids})
            for eps in eps_list:
                db = DBSCAN(
                    eps=eps,
                    min_samples=args.dbscan_min_samples,
                    metric="euclidean"
                )
                labels = db.fit_predict(embedding_20d)
                db_clusters_dict[eps] = (labels, None)
                sweep_df[f"eps_{eps:.3f}"] = labels
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"DBSCAN eps={eps}: {n_clusters} clusters")

            sweep_df.to_csv(out_path("dbscan_sweep.tsv"), sep="\t", index=False)
            print("DBSCAN sweep clusters saved")

            # last eps as reference
            last_eps = eps_list[-1]
            db_labels = db_clusters_dict[last_eps][0]  # <- extract labels only

        else:
            print(f"Running DBSCAN eps={args.dbscan_eps}...")
            db = DBSCAN(
                eps=args.dbscan_eps,
                min_samples=args.dbscan_min_samples,
                metric="euclidean"
            )
            db_labels = db.fit_predict(embedding_20d)
            pd.DataFrame({"id": ids, "cluster": db_labels}).to_csv(out_path("dbscan_clusters.tsv"), sep="\t", index=False)
            print("DBSCAN clustering saved")

    # -----------------------------------------------------
    # 2D Visualization
    # -----------------------------------------------------
    if args.visualize:
        print("Computing 2D UMAP for visualization...")
        reducer_2d = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=0.3,
            n_components=2,
            metric=args.metric,
            random_state=args.seed
        )
        embedding_2d = reducer_2d.fit_transform(X)
        pd.DataFrame(embedding_2d, columns=["UMAP_1", "UMAP_2"]).to_csv(out_path("embedding_2d.tsv"), sep="\t", index=False)

        # Plot Leiden
        plot_embedding(
            embedding_2d,
            last_labels,
            output_dir=args.output,
            save_name=f"{prefix}_umap_leiden.png",
            title="Leiden on UMAP graph"
        )

        # Plot DBSCAN last eps
        if db_labels is not None:
            plot_embedding(
                embedding_2d,
                db_labels,
                output_dir=args.output,
                save_name=f"{prefix}_umap_dbscan.png",
                title=f"DBSCAN"
            )

        # Sweep grid plots
        if args.resolution_sweep:
            plot_embedding_grid(
                embedding_2d,
                clusters_dict,
                output_dir=args.output,
                save_name=f"{prefix}_umap_leiden_sweep.png"
            )
        if args.dbscan_sweep and len(db_clusters_dict) > 1:
            plot_embedding_grid(
                embedding_2d,
                db_clusters_dict,
                output_dir=args.output,
                save_name=f"{prefix}_umap_dbscan_sweep.png"
            )

    print("Done.")


if __name__ == "__main__":
    main()
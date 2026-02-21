#!/usr/bin/env python

import argparse
import os
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

from knn_graph_leiden.clustering import run_leiden, leiden_resolution_sweep
from knn_graph_leiden.visualisation import (
    plot_embedding,
    plot_embedding_grid
)
from knn_graph_leiden.validation import load_ground_truth, evaluate_ground_truth
from knn_graph_leiden.utils.timing import PipelineTracker
from knn_graph_leiden.utils.logging_utils import section, info, warn

# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="UMAP-topology-based clustering using Leiden or DBSCAN with optional sweeps"
    )

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # UMAP parameters
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_dim", type=int, default=20)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--l2norm", action="store_true")

    # Leiden
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--resolution_sweep", action="store_true")

    # DBSCAN
    parser.add_argument("--dbscan", action="store_true")
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=10)
    parser.add_argument("--dbscan_sweep", action="store_true")

    # Visualization / evaluation
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--embedding_tsv", type=str, default=None)
    parser.add_argument("--ground_truth", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.index.values, df.values.astype(np.float32)


# ---------------------------------------------------------
def main():
    print("-" * 70)
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    def out_path(fname):
        return os.path.join(args.output, f"{prefix}_{fname}")

    tracker = PipelineTracker(verbose=True)

    # ---------------------------------------------------------
    section("Data loading")
    tracker.start("load_data")
    ids, X = load_data(args.input)
    tracker.stop()

    if args.l2norm:
        section("L2 normalization")
        tracker.start("l2_normalization")
        X = normalize(X)
        tracker.stop()

    # ---------------------------------------------------------
    section("UMAP topology inference")
    tracker.start("umap_high_dim")
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_dim,
        metric=args.metric,
        random_state=args.seed
    )
    embedding_hd = reducer.fit_transform(X)
    graph = reducer.graph_
    tracker.stop()
    info(f"UMAP graph edges: {graph.nnz}")

    # ---------------------------------------------------------
    section("Leiden clustering")
    clusters_dict = {}
    last_labels = None

    if args.resolution_sweep:
        resolutions = [
            0.001, 0.0025, 0.005, 0.0075,
            0.01, 0.025, 0.05, 0.075,
            0.1, 0.25, 0.5, 0.75,
            1.0, 2.5, 5.0
        ]
        sweep_df = pd.DataFrame({"id": ids})

        for r in resolutions:
            stage_name = f"leiden_r_{r}"
            tracker.start(stage_name)
            labels, modularity = run_leiden(graph, resolution=r, seed=args.seed)
            tracker.stop()
            clusters_dict[r] = (labels, modularity)
            sweep_df[f"r_{r:.4f}"] = labels

        sweep_df.to_csv(out_path("clusters_sweep.tsv"), sep="\t", index=False)
        info("Leiden sweep clusters saved")
        last_labels = clusters_dict[resolutions[-1]][0]

    else:
        tracker.start("leiden_single")
        labels, modularity = run_leiden(graph, resolution=args.resolution, seed=args.seed)
        tracker.stop()
        clusters_dict[args.resolution] = (labels, modularity)
        last_labels = labels
        pd.DataFrame({"id": ids, "cluster": labels}).to_csv(
            out_path("leiden_clusters.tsv"), sep="\t", index=False
        )
        info("Leiden clustering saved")

    # ---------------------------------------------------------
    section("DBSCAN clustering")
    db_labels = None
    db_clusters_dict = {}

    if args.dbscan:
        if args.dbscan_sweep:
            eps_list = [
                0.01, 0.025, 0.05, 0.075,
                0.1, 0.25, 0.5, 0.75,
                1.0, 2.5, 5.0
            ]
            sweep_df = pd.DataFrame({"id": ids})

            for eps in eps_list:
                stage_name = f"dbscan_eps_{eps}"
                tracker.start(stage_name)
                db = DBSCAN(
                    eps=eps, min_samples=args.dbscan_min_samples, metric="euclidean"
                )
                labels = db.fit_predict(embedding_hd)
                tracker.stop()

                db_clusters_dict[eps] = (labels, None)
                sweep_df[f"eps_{eps:.3f}"] = labels
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                info(f"DBSCAN eps={eps}: {n_clusters} clusters")

            sweep_df.to_csv(out_path("dbscan_sweep.tsv"), sep="\t", index=False)
            info("DBSCAN sweep clusters saved")
            db_labels = db_clusters_dict[eps_list[-1]][0]

        else:
            tracker.start("dbscan_single")
            db = DBSCAN(
                eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, metric="euclidean"
            )
            db_labels = db.fit_predict(embedding_hd)
            tracker.stop()

            pd.DataFrame({"id": ids, "cluster": db_labels}).to_csv(
                out_path("dbscan_clusters.tsv"), sep="\t", index=False
            )
            info("DBSCAN clustering saved")

    # ---------------------------------------------------------
    section("Ground truth evaluation")
    if args.ground_truth and last_labels is not None:
        tracker.start("ground_truth_evaluation")
        gt_dict = load_ground_truth(args.ground_truth)
        gt_metrics = evaluate_ground_truth(last_labels, ids, gt_dict)
        pd.DataFrame([gt_metrics]).to_csv(
            out_path("ground_truth_metrics.tsv"), sep="\t", index=False
        )
        tracker.stop()
        info("Ground-truth metrics saved")

    # ---------------------------------------------------------
    section("2D visualization")
    if args.visualize:
        emb_path = out_path("embedding_2d.tsv")

        if args.embedding_tsv:
            info("Loading provided embedding")
            embedding_2d = pd.read_csv(args.embedding_tsv, sep="\t").values
        elif os.path.exists(emb_path):
            info("Reusing cached embedding")
            embedding_2d = pd.read_csv(emb_path, sep="\t").values
        else:
            tracker.start("umap_2d_visualization")
            reducer_2d = umap.UMAP(
                n_neighbors=args.umap_neighbors, min_dist=0.3, n_components=2,
                metric=args.metric, random_state=args.seed
            )
            embedding_2d = reducer_2d.fit_transform(X)
            pd.DataFrame(embedding_2d, columns=["UMAP_1", "UMAP_2"]).to_csv(
                emb_path, sep="\t", index=False
            )
            tracker.stop()
            info("2D embedding saved")

        plot_embedding(embedding_2d, last_labels, output_dir=args.output,
                       save_name=f"{prefix}_umap_leiden.png", title="Leiden on UMAP graph")

        if db_labels is not None:
            plot_embedding(embedding_2d, db_labels, output_dir=args.output,
                           save_name=f"{prefix}_umap_dbscan.png", title="DBSCAN")

        if args.resolution_sweep:
            plot_embedding_grid(embedding_2d, clusters_dict, output_dir=args.output,
                                save_name=f"{prefix}_umap_leiden_sweep.png")

        if args.dbscan_sweep and len(db_clusters_dict) > 1:
            plot_embedding_grid(embedding_2d, db_clusters_dict, output_dir=args.output,
                                save_name=f"{prefix}_umap_dbscan_sweep.png")

    # ---------------------------------------------------------
    section("Saving runtime metrics")
    tracker.save(out_path("runtime.tsv"))
    section("done")
    print("-" * 70)


if __name__ == "__main__":
    main()
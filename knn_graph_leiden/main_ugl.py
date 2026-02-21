#!/usr/bin/env python

import argparse
import os
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from scipy import sparse
import igraph as ig

from knn_graph_leiden.clustering import run_leiden, leiden_resolution_sweep
from knn_graph_leiden.visualisation import plot_embedding, plot_embedding_grid
from knn_graph_leiden.validation import load_ground_truth, evaluate_ground_truth
from knn_graph_leiden.utils.timing import PipelineTracker
from knn_graph_leiden.utils.logging_utils import section, info, warn
from knn_graph_leiden.io import load_input
from knn_graph_leiden.metrics import evaluate_clustering

# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=("UMAP-based topology inference followed by clustering using "
                     "Leiden (graph-based) and/or DBSCAN (embedding-based), "
                     "with optional parameter sweeps and visualization."))

    parser.add_argument("-i", "--input", required=True,
                        help="Input data file (.tsv with index column OR .npy array; rows=samples, cols=features)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for clusters, metrics, runtime logs, and plots")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Optional prefix for all output files (default: derived from input filename)")

    # UMAP parameters
    parser.add_argument("--umap_neighbors", type=int, default=30,
                        help="Number of neighbors for UMAP graph construction (n_neighbors)")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter controlling embedding compactness")
    parser.add_argument("--umap_dim", type=int, default=20,
                        help="Dimensionality of UMAP embedding used for clustering")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="Distance metric used by UMAP")
    parser.add_argument("--l2norm", action="store_true",
                        help="Apply L2 normalization before UMAP (recommended for cosine metric)")

    # Leiden clustering
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                        help="Leiden resolution parameter (ignored if --resolution_sweep is used)")
    parser.add_argument("--resolution_sweep", action="store_true",
                        help="Perform Leiden resolution sweep over predefined grid")

    # DBSCAN clustering
    parser.add_argument("--dbscan", action="store_true",
                        help="Run DBSCAN clustering on UMAP embedding")
    parser.add_argument("--dbscan_eps", type=float, default=0.5,
                        help="DBSCAN epsilon (radius parameter; ignored if --dbscan_sweep is used)")
    parser.add_argument("--dbscan_min_samples", type=int, default=10,
                        help="Minimum samples for DBSCAN core points")
    parser.add_argument("--dbscan_sweep", action="store_true",
                        help="Perform DBSCAN epsilon sweep over predefined grid")

    # Visualization / evaluation
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 2D UMAP visualization of clustering results")
    parser.add_argument("--embedding_tsv", type=str, default=None,
                        help="Optional precomputed 2D embedding TSV (2 columns, same row order as input)")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="TSV file with ground-truth labels (id<TAB>cluster) for external evaluation")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for UMAP and clustering")
    return parser.parse_args()

def csr_to_igraph(csr):
    sources, targets = csr.nonzero()
    weights = csr.data
    g = ig.Graph(n=csr.shape[0], edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights
    return g

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
    ids, X = load_input(args.input)
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
    igraph_graph = csr_to_igraph(graph)
    tracker.stop()
    info(f"UMAP graph edges: {graph.nnz}")

    # Save high-dimensional embedding used for clustering
    section("save UMAP embedding and graph")
    tracker.start("umap_save")
    embedding_hd_path = out_path("embedding_umap_hd.tsv")
    pd.DataFrame(embedding_hd, index=ids, columns=[f"UMAP_{i+1}" for i in range(embedding_hd.shape[1])]).to_csv(embedding_hd_path, sep="\t")
    info("UMAP embedding (clustering space) saved")
    graph_path = out_path("umap_graph.npz")
    sparse.save_npz(graph_path, graph)
    info("UMAP graph (sparse CSR) saved")
    #import joblib
    #joblib.dump(reducer, out_path("umap_model.joblib"))
    #info("saved umap model as joblib")
    tracker.stop()


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

    section("Leiden cluster evaluation")

    metrics_list = []
    for res, (labels, modularity) in clusters_dict.items():
        if labels.size > 0:
            tracker.start(f"metrics_r_{res}")
            metrics = evaluate_clustering(X, igraph_graph, labels)
            tracker.stop()
            metrics["resolution"] = res
            metrics_list.append(metrics)
            info(f"Leiden, r={res}, {len(np.unique(labels))} clusters")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_path("leiden_metrics.tsv"), sep="\t", index=False)

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
                info(f"DBSCAN, eps={eps}, {n_clusters} clusters")

            sweep_df.to_csv(out_path("dbscan_sweep.tsv"), sep="\t", index=False)
            info("DBSCAN sweep clusters saved")
            db_labels = db_clusters_dict[eps_list[-1]][0]
            
            section("DBSCAN evaluation")
            db_metrics_list = []
            for eps, (labels, _) in db_clusters_dict.items():
                tracker.start(f"metrics_dbscan_eps_{eps}")
                metrics = evaluate_clustering(embedding_hd, None, labels)
                tracker.stop()
                metrics["eps"] = eps
                db_metrics_list.append(metrics)

            db_metrics_df = pd.DataFrame(db_metrics_list)
            db_metrics_df.to_csv(out_path("dbscan_metrics.tsv"), sep="\t", index=False)
            info("DBSCAN metrics saved")
                                 

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
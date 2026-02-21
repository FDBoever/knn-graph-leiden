#!/usr/bin/env python

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import umap
import igraph as ig

from knn_graph_leiden.knn import build_knn
from knn_graph_leiden.graph import build_graph, refine_snn, combine_weights, to_igraph
from knn_graph_leiden.clustering import run_leiden, leiden_resolution_sweep
from knn_graph_leiden.metrics import evaluate_clustering
from knn_graph_leiden.visualisation import plot_embedding, generate_summary_plots, plot_embedding_grid, plot_resolution_metrics
from knn_graph_leiden.stability import detect_main_clusters, merge_small_clusters
from knn_graph_leiden.selection import recommend_resolution
from knn_graph_leiden.stitching import compute_cluster_connectivity, find_merges, merge_clusters
from knn_graph_leiden.io import load_input
from knn_graph_leiden.utils.timing import PipelineTracker
from knn_graph_leiden.utils.logging_utils import section, info, warn

# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="kNN graph-based clustering using Leiden algorithm, with optional approximate kNN"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input .tsv file or .npy (rows=samples, cols=features)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for clusters, metrics, and plots")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Optional prefix for all output files")
    parser.add_argument("-k", "--neighbors", type=int, default=30,
                        help="Number of nearest neighbors for kNN")
    parser.add_argument("-m", "--metric", choices=["cosine", "euclidean"],
                        default="euclidean", help="Distance metric for kNN")
    parser.add_argument("--l2norm", action="store_true",
                        help="Apply L2 normalization before kNN")
    parser.add_argument("--mutual", action="store_true",
                        help="Use mutual kNN")
    parser.add_argument("--snn", action="store_true",
                        help="Apply SNN refinement")
    parser.add_argument("--tanimoto", action="store_true",
                        help="Apply Tanimoto coefficient refinement")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Weighting for combining kNN and SNN graphs")
    parser.add_argument("--prune", type=float, default=0.001,
                        help="Pruning threshold for graph edges")
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                        help="Leiden resolution parameter")
    parser.add_argument("--sweep", action="store_true",
                        help="Perform resolution sweep (0.001-1.0)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate UMAP visualization of clusters")
    parser.add_argument("--embedding_tsv", type=str, default=None,
                        help="Optional 2D embedding TSV file (2 columns, same row order as input)")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="TSV file with ground truth clusters (id\tcluster)")
    parser.add_argument("--knn_method", choices=["auto","exact","hnsw"], default="auto",
                        help="kNN method: 'auto'=large dataset uses HNSW, 'exact', or 'hnsw'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

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
    section("kNN graph construction")
    tracker.start("knn_build")
    hnsw_kwargs = {"ef": 200, "M": 16, "ef_query": 100, "random_seed": args.seed}
    labels_knn, distances = build_knn(
        X, k=args.neighbors, metric=args.metric, method=args.knn_method,
        hnsw_kwargs=hnsw_kwargs, n_jobs=-1
    )
    tracker.stop()

    # ---------------------------------------------------------
    section("Graph construction")
    tracker.start("graph_build")
    A = build_graph(
        labels_knn, distances, X, metric=args.metric,
        mutual=args.mutual, prune_threshold=args.prune,
        tanimoto=args.tanimoto
    )
    tracker.stop()
    info(f"sparse edges: {A.nnz}")

    del labels_knn, distances

    # ---------------------------------------------------------
    if args.snn and A.nnz > 0:
        section("SNN refinement")
        tracker.start("snn_refinement")
        snn = refine_snn(A)
        if snn.nnz > 0:
            A = combine_weights(A, snn, alpha=args.alpha, prune_threshold=args.prune)
        tracker.stop()
        info(f"edges after SNN: {A.nnz}")

    # ---------------------------------------------------------
    section("igraph construction")
    tracker.start("igraph_conversion")
    g = to_igraph(A) if A.nnz > 0 else None
    tracker.stop()
    if g is not None:
        info(f"nodes={g.vcount()}, edges={g.ecount()}")
    else:
        warn("graph has no edges")

    # ---------------------------------------------------------
    section("Leiden clustering")
    clusters_dict = {}
    if A.nnz > 0:
        if args.sweep:
            resolutions = [
                0.001, 0.0025, 0.005, 0.0075,
                0.01, 0.025, 0.05, 0.075,
                0.1, 0.25, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5
            ]
            for r in resolutions:
                tracker.start(f"leiden_r_{r}")
                labels, modularity = run_leiden(A, resolution=r, seed=args.seed)
                tracker.stop()
                clusters_dict[r] = (labels, modularity)
        else:
            tracker.start("leiden_single")
            labels, modularity = run_leiden(A, resolution=args.resolution, seed=args.seed)
            tracker.stop()
            clusters_dict[args.resolution] = (labels, modularity)
    else:
        clusters_dict[args.resolution] = (np.array([]), np.nan)
        warn("skipping clustering (no edges)")

    # ---------------------------------------------------------
    section("Cluster evaluation")
    metrics_list = []
    for res, (labels, modularity) in clusters_dict.items():
        if labels.size > 0:
            tracker.start(f"metrics_r_{res}")
            metrics = evaluate_clustering(X, g, labels)
            tracker.stop()
            metrics["resolution"] = res
            metrics_list.append(metrics)
            info(f"Leiden, r={res},{len(np.unique(labels))} clusters")
            

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_path("metrics.tsv"), sep="\t", index=False)

    # ---------------------------------------------------------
    # Save sweep clusters
    if args.sweep:
        section("Saving sweep clusters")
        sweep_df = pd.DataFrame({"id": ids})
        for res in sorted(clusters_dict.keys()):
            labels, _ = clusters_dict[res]
            if labels.size > 0:
                sweep_df[f"r_{res:.4f}"] = labels
        sweep_df.to_csv(out_path("clusters_sweep.tsv"), sep="\t", index=False)
        last_labels = clusters_dict[sorted(clusters_dict.keys())[0]][0]
    else:
        last_labels = list(clusters_dict.values())[-1][0]
        if last_labels.size > 0:
            pd.DataFrame({"id": ids, "cluster": last_labels}).to_csv(
                out_path("clusters.tsv"), sep="\t", index=False
            )

    # ---------------------------------------------------------
    section("Cluster stitching")
    tracker.start("cluster_stitching")
    clusters, intra, inter = compute_cluster_connectivity(A, last_labels)
    merges = find_merges(clusters, intra, inter, threshold=0.05)
    consolidated_labels = merge_clusters(last_labels, merges) if merges else last_labels.copy()
    tracker.stop()
    pd.DataFrame({
        "id": ids,
        "consolidated_cluster": consolidated_labels,
        "original_cluster": last_labels
    }).to_csv(out_path("clusters_consolidated.tsv"), sep="\t", index=False)

    # ---------------------------------------------------------
    if args.visualize and last_labels.size > 0:
        section("UMAP visualization")
        emb_path = out_path("embedding.tsv")
        if args.embedding_tsv:
            embedding = pd.read_csv(args.embedding_tsv, sep="\t").values
        elif os.path.exists(emb_path):
            embedding = pd.read_csv(emb_path, sep="\t").values
        else:
            tracker.start("umap_2d")
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric=args.metric, random_state=args.seed)
            embedding = reducer.fit_transform(X)
            tracker.stop()
            pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"]).to_csv(emb_path, sep="\t", index=False)

        # -- Basic embedding plots --
        plot_embedding(embedding, last_labels, output_dir=args.output,
                       save_name=f"{prefix}_embedding_original_clusters.png",
                       title="Leiden clusters")
        plot_embedding(embedding, consolidated_labels, output_dir=args.output,
                       save_name=f"{prefix}_consolidated_clusters.png",
                       title="Merged clusters")

        # -- Advanced summary plots --
        generate_summary_plots(embedding, last_labels, adjacency_matrix=A, output_dir=args.output, prefix=prefix)
        if args.sweep and len(clusters_dict) > 1:
            plot_embedding_grid(embedding, clusters_dict, output_dir=args.output,
                                save_name=f"{prefix}_leiden_sweep.png")
        if not metrics_df.empty:
            plot_resolution_metrics(metrics_df, output_dir=args.output, prefix=f"{prefix}_metrics")

    # ---------------------------------------------------------
    section("Saving runtime metrics")
    tracker.save(out_path("runtime.tsv"))
    section("Pipeline finished")
    print("-" * 70)


if __name__ == "__main__":
    main()
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

# formatted printing
def section(title):
    print(f"{title}")

def info(msg):
    print(f"   - {msg}")

def warn(msg):
    print(f"  !!! {msg}")


# argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="kNN graph-based clustering using Leiden algorithm, with optional approximate kNN"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input TSV file (rows=samples, cols=features)")
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


# load table
def load_data(path):
    section("Loading data")
    df = pd.read_csv(path, sep="\t", index_col=0)
    info(f"Data loaded: {df.shape[0]} samples x {df.shape[1]} features")
    return df.index.values, df.values.astype(np.float32)


# main pipeline
# ---------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    def out_path(fname):
        return os.path.join(args.output, f"{prefix}_{fname}")

    # Load data
    ids, X = load_data(args.input)
    if args.l2norm:
        info("Applying L2 normalization...")
        X = normalize(X)

    # kNN graph construction
    section("kNN graph construction")
    info(f"kNN parameters: k={args.neighbors}, metric={args.metric}, method={args.knn_method}")
    hnsw_kwargs = {"ef": 200, "M": 16, "ef_query": 100, "random_seed": args.seed}
    labels_knn, distances = build_knn(
        X, k=args.neighbors, metric=args.metric, method=args.knn_method, hnsw_kwargs=hnsw_kwargs, n_jobs=-1
    )
    info(f"kNN graph built: {labels_knn.shape[0]} samples with {labels_knn.shape[1]} neighbors each")

    # Build graph
    A = build_graph(
        labels_knn, distances, X,
        metric=args.metric,
        mutual=args.mutual,
        prune_threshold=args.prune,
        tanimoto=args.tanimoto
    )

    info(f"graph constructed: {A.nnz} edges (sparse)")
    del labels_knn
    del distances

    # SNN refinement
    if args.snn and A.nnz > 0:
        info("Applying SNN refinement...")
        snn = refine_snn(A)
        if snn.nnz > 0:
            A = combine_weights(A, snn, alpha=args.alpha, prune_threshold=args.prune)
            info(f"SNN combined: {A.nnz} edges after refinement")
        else:
            warn("SNN graph is empty. Skipping combination.")

    # Build igraph object efficiently
    g = to_igraph(A) if A.nnz > 0 else None
    if g is None:
        warn("graph has no edges; igraph object is None")
    else:
        info(f"igraph object created with {g.vcount()} nodes and {g.ecount()} edges")

    # Leiden clustering
    section("clustering with Leiden algorithm")
    clusters_dict = {}
    if A.nnz > 0:
        if args.sweep:
            info("Performing resolution sweep")
            resolutions = [0.001, 0.0025, 0.005, 0.0075,
                           0.01, 0.025, 0.05, 0.075, 
                           0.1, 0.25, 0.5, 0.75, 1.0,
                           1, 2.5, 5, 7.5]
            clusters_dict = leiden_resolution_sweep(A, resolutions=resolutions, seed=args.seed)
        else:
            labels, modularity = run_leiden(A, resolution=args.resolution, seed=args.seed)
            clusters_dict[args.resolution] = (labels, modularity)
    else:
        clusters_dict[args.resolution] = (np.array([]), np.nan)
        warn("graph has no edges. Skipping clustering.")

    # evaluate metrics
    section("Evaluating clusters")
    metrics_list = []
    for res, (labels, modularity) in clusters_dict.items():
        if labels.size > 0:
            metrics = evaluate_clustering(X, g, labels)
            metrics["resolution"] = res
            metrics_list.append(metrics)
            info(f"Resolution {res}: modularity={modularity:.4f}, n_clusters={len(np.unique(labels))}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_path("metrics.tsv"), sep="\t", index=False)
    info(f"clustering metrics saved to {out_path('metrics.tsv')}")

    if args.sweep:
        section("Saving sweep cluster labels")
        sweep_df = pd.DataFrame({"id": ids})

        # Sort resolutions to ensure order
        sorted_res = sorted(clusters_dict.keys())

        for res in sorted_res:
            labels, _ = clusters_dict[res]
            if labels.size > 0:
                col_name = f"r_{res:.4f}"
                sweep_df[col_name] = labels

        sweep_df.to_csv(out_path("clusters_sweep.tsv"), sep="\t", index=False)
        info(f"sweep cluster labels saved: {out_path('clusters_sweep.tsv')}")

        #define reference clustering (highest resolution)
        last_res = sorted_res[0]
        last_labels = clusters_dict[last_res][0]

    else:
        last_labels = list(clusters_dict.values())[-1][0]
        if last_labels.size > 0:
            pd.DataFrame({"id": ids, "cluster": last_labels}).to_csv(
                out_path("clusters.tsv"), sep="\t", index=False
            )
            info(f"cluster labels saved: {out_path('clusters.tsv')}")  
    
    # Consolidate clusters
    section("Consolidating clusters")
    #main_clusters = detect_main_clusters(last_labels, top_fraction=0.99)
    #consolidated_labels, original_labels = merge_small_clusters(X, last_labels, main_clusters)
    section("Stitching over-segmented clusters")

    clusters, intra, inter = compute_cluster_connectivity(A, last_labels)

    merges = find_merges(clusters, intra, inter, threshold=0.05)

    if merges:
        consolidated_labels = merge_clusters(last_labels, merges)
    else:
        consolidated_labels = last_labels.copy()
    original_labels = last_labels

    pd.DataFrame({
        "id": ids,
        "consolidated_cluster": consolidated_labels,
        "original_cluster": original_labels
    }).to_csv(out_path("clusters_consolidated.tsv"), sep="\t", index=False)
    info(f"consolidated clusters saved: {out_path('clusters_consolidated.tsv')}")

    # UMAP embedding + visualization
    if args.visualize and last_labels.size > 0:
        section("UMAP embedding")
        embedding = None
        emb_path = out_path("embedding.tsv")

        if args.embedding_tsv:
            info("loading provided embedding")
            embedding = pd.read_csv(args.embedding_tsv, sep="\t").values
        elif os.path.exists(emb_path):
            info("reusing cached embedding")
            embedding = pd.read_csv(emb_path, sep="\t").values
        else:
            info("UMAP embedding...")
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric=args.metric, random_state=args.seed)
            embedding = reducer.fit_transform(X)
            pd.DataFrame(embedding, columns=["UMAP_1","UMAP_2"]).to_csv(emb_path, sep="\t", index=False)
            info(f"UMAP embedding saved: {emb_path}")

        plot_embedding(embedding, last_labels, output_dir=args.output,
                       save_name=f"{prefix}_embedding_original_clusters.png",
                       title="Leiden clusters")
        plot_embedding(embedding, consolidated_labels, output_dir=args.output,
                       save_name=f"{prefix}_consolidated_clusters.png",
                       title="merged clusters")
        info("plots saved")
    
    #remove from memory
    del X

    if args.sweep and not metrics_df.empty:
        recommended_r, scored_df = recommend_resolution(metrics_df)
        info(f"Recommended resolution: {recommended_r}")
        scored_df.to_csv(out_path("resolution_scored.tsv"), sep="\t", index=False)

    if args.sweep and args.visualize:
        plot_embedding_grid(
            embedding,
            clusters_dict,
            output_dir=args.output,
            save_name=f"{prefix}_embedding_sweep.png"
        )

    if args.visualize and last_labels.size > 0:
        generate_summary_plots(
            embedding=embedding,
            labels=consolidated_labels,
            adjacency_matrix=A,
            output_dir=args.output,
            prefix=prefix
        )
    
    if args.sweep and not metrics_df.empty:
        plot_resolution_metrics(
            metrics_df,
            output_dir=args.output,
            prefix=prefix
        )

    # Ground-truth evaluation
    if args.ground_truth and last_labels.size > 0:
        section("ground-truth evaluation")
        from graph_pipeline.validation import load_ground_truth, evaluate_ground_truth
        gt_dict = load_ground_truth(args.ground_truth)
        gt_metrics_orig = evaluate_ground_truth(last_labels, ids, gt_dict)
        gt_metrics_consol = evaluate_ground_truth(consolidated_labels, ids, gt_dict)
        pd.DataFrame([gt_metrics_orig, gt_metrics_consol], index=["original","consolidated"]).to_csv(
            out_path("ground_truth_metrics.tsv"), sep="\t")
        info("ground-truth metrics saved")
    
    print("-"*80)
    print("done...")


if __name__ == "__main__":
    main()
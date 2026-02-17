# knn_graph_leiden/__init__.py
# expose functions at the package level
from .graph import build_graph, refine_snn, combine_weights, to_igraph
from .clustering import run_leiden, leiden_resolution_sweep
from .metrics import evaluate_clustering
from .visualization import plot_embedding, generate_summary_plots, plot_embedding_grid, plot_resolution_metrics
from .stability import detect_main_clusters, merge_small_clusters
from .selection import recommend_resolution

__version__ = "0.1.0"

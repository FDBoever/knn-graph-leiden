import numpy as np
import scipy.sparse as sp
import igraph as ig


# ---------------------------------------------------------
# Build weighted kNN graph (memory-safe)
# ---------------------------------------------------------
def build_graph(
    labels,
    distances,
    data_matrix=None,
    metric="cosine",
    mutual=False,
    prune_threshold=1e-3,
    tanimoto=False,
):
    """
    Memory-efficient kNN graph builder.

    Returns:
        CSR sparse symmetric adjacency matrix (float32)
    """

    n, k = labels.shape

    # Use int32 to reduce memory
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = labels.astype(np.int32).ravel()

    # --------------------
    # Distance → similarity
    # --------------------
    if metric == "cosine":
        vals = (1.0 - distances).ravel()
    else:
        sigma = np.median(distances)
        sigma = sigma if sigma > 0 else 1e-12
        vals = np.exp(-(distances ** 2) / (2 * sigma ** 2)).ravel()

    vals = vals.astype(np.float32)


    # --------------------
    # Optional Tanimoto (chunked, memory-friendly)
    # --------------------
    if tanimoto and data_matrix is not None:

        data_matrix = data_matrix.astype(np.float32, copy=False)
        norms_sq = np.sum(data_matrix ** 2, axis=1).astype(np.float32)

        # avoid huge temporary matrices
        chunk_size = 50000  # adjust if needed (safe for 15k+)

        for start in range(0, len(rows), chunk_size):
            end = min(start + chunk_size, len(rows))

            i_chunk = rows[start:end]
            j_chunk = cols[start:end]

            # Only small temporary matrix allocated here
            dots = np.sum(
                data_matrix[i_chunk] * data_matrix[j_chunk],
                axis=1
            )

            denom = norms_sq[i_chunk] + norms_sq[j_chunk] - dots

            mask = denom > 0

            vals[start:end][mask] *= dots[mask] / denom[mask]
            vals[start:end][~mask] = 0.0

    # --------------------
    # Build directed sparse adjacency
    # --------------------
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

    # Convert immediately to CSR (more memory stable)
    A = A.tocsr()

    # --------------------
    # Symmetrize safely
    # --------------------
    if mutual:
        A = A.minimum(A.transpose())
    else:
        A = A.maximum(A.transpose())

    # --------------------
    # Remove self-loops
    # --------------------
    A.setdiag(0)
    A.eliminate_zeros()

    # --------------------
    # Prune
    # --------------------
    if prune_threshold > 0 and A.nnz > 0:
        A.data[A.data < prune_threshold] = 0
        A.eliminate_zeros()

    # --------------------
    # Normalize
    # --------------------
    if A.nnz > 0:
        A.data /= A.data.max()

    return A


# ---------------------------------------------------------
# Shared Nearest Neighbor (SNN)
# ---------------------------------------------------------
def refine_snn(A):
    """
    Sparse SNN refinement.
    Safe for large graphs.
    """

    if A.nnz == 0:
        return sp.csr_matrix(A.shape, dtype=np.float32)

    # Binary connectivity (cheap)
    B = A.copy()
    B.data[:] = 1.0

    # Sparse multiplication
    snn = B @ B.T
    snn.setdiag(0)
    snn.eliminate_zeros()

    if snn.nnz > 0:
        snn.data = snn.data.astype(np.float32)
        snn.data /= snn.data.max()

    return snn


# ---------------------------------------------------------
# Combine kNN + SNN
# ---------------------------------------------------------
def combine_weights(A, snn, alpha=0.8, prune_threshold=1e-3):
    """
    Combine two sparse graphs safely.
    """

    if snn is None or snn.nnz == 0:
        return A

    final = alpha * A + (1 - alpha) * snn

    # Symmetry safety
    final = final.maximum(final.transpose())

    final.setdiag(0)
    final.eliminate_zeros()

    if prune_threshold > 0 and final.nnz > 0:
        final.data[final.data < prune_threshold] = 0
        final.eliminate_zeros()

    if final.nnz > 0:
        final.data /= final.data.max()

    return final


# ---------------------------------------------------------
# Sparse → igraph conversion (NO Python tuples)
# ---------------------------------------------------------
def to_igraph(A):
    """
    Ultra memory-efficient igraph conversion.
    Uses NumPy edges (no Python tuples).
    Upper triangle only.
    """

    if A is None or A.nnz == 0:
        return None

    A = A.tocoo()

    # Upper triangle mask
    mask = A.row < A.col

    if not np.any(mask):
        return None

    # Use int32 for minimal memory
    edges = np.column_stack((
        A.row[mask].astype(np.int32),
        A.col[mask].astype(np.int32),
    ))

    weights = A.data[mask].astype(np.float32)

    g = ig.Graph(n=A.shape[0], directed=False)
    g.add_edges(edges)
    g.es["weight"] = weights

    return g

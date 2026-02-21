import pandas as pd
import numpy as np
import os

def load_input(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t", index_col=0)
        return df.index.to_numpy(), df.values.astype(np.float32)

    elif ext == ".npy":
        X = np.load(path).astype(np.float32)
        if X.ndim != 2:
            raise ValueError("NPY input must be a 2D array")
        ids = np.array([f"sample_{i}" for i in range(X.shape[0])])
        return ids, X

    else:
        raise ValueError(f"Unsupported input format: {ext}")
    
    
def load_clr_tsv(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.index.to_numpy(), df.values.astype(np.float32)


def save_clusters(ids, labels, path):
    df = pd.DataFrame({
        "id": ids,
        "cluster": labels
    })
    df.to_csv(path, sep="\t", index=False)


def save_metrics(metrics_dict, path):
    pd.DataFrame(metrics_dict).to_csv(path, sep="\t", index=False)


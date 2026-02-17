import pandas as pd
import numpy as np


def load_clr_tsv(path):
    """
    Load CLR-transformed TSV.
    Rows = samples, columns = features.
    """
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


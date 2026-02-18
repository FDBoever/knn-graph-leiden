import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t")
    if not {"id","cluster"}.issubset(df.columns):
        raise ValueError("Ground truth TSV must contain columns 'id' and 'cluster'")
    return dict(zip(df["id"], df["cluster"]))

def evaluate_ground_truth(pred_labels, ids, gt_dict):
    # Keep only IDs present in ground truth
    filtered_indices = [i for i, id_ in enumerate(ids) if id_ in gt_dict]
    if len(filtered_indices) == 0:
        raise ValueError("No IDs from prediction found in ground truth")

    pred = pred_labels[filtered_indices]
    true = [gt_dict[ids[i]] for i in filtered_indices]

    metrics = {
        "ARI": adjusted_rand_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
        "FMI": fowlkes_mallows_score(true, pred)
    }
    return metrics
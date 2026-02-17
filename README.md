# knn-graph-leiden
kNN graph-based clustering pipeline using the Leiden algorithm, with optional SNN refinement and resolution sweeps for high-dimensional data.

## usage example

```sh
python -m knn_graph_leiden.main \
    -i subset_5000.tsv \
    -o results2 \
    -k 50 \
    --metric euclidean \
    --l2norm \
    --tanimoto \
    -r 0.01 \
    --visualize \
    --knn_method hnsw \
    --sweep \
    --snn
```

```sh
kgl -i subset_5000.tsv \
    -o results2 \
    -k 50 \
    --metric euclidean \
    --l2norm \
    --tanimoto \
    -r 0.01 \
    --visualize \
    --knn_method hnsw \
    --sweep \
    --snn
```


## Conda if required

Create `environment.yml`:

```yaml
name: knn-graph-leiden
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - seaborn
  - matplotlib
  - umap-learn
  - python-igraph
  - leidenalg
  - hnswlib
  - pip
```

Then run:

```bash
conda env create -f environment.yml
conda activate knn-graph-leiden
```


# Input

## Feature Matrix (Required)

Requires tab separated file (tsp), where rows are samples, and column are features
The first column stores the sample_id's or index, and the first row column names


```
        feat1   feat2   feat3
seq0    0.12    1.04    -0.44
seq1    0.18    0.95    -0.38
```

---

## ground truth labels

```
id      cluster
seq0    0
seq1    0
seq2    1
```
from setuptools import setup, find_packages

setup(
    name="knn-graph-leiden",
    version="0.1.0",
    description="kNN graph-based clustering with Leiden algorithm and visualisation",
    author="FDB",
    author_email="fdb@mail",
    packages=find_packages(),  # automatically finds knn_graph_leiden
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "igraph",
        "hnswlib",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "seaborn"
    ],
    entry_points={
        "console_scripts": [
            "kgl=knn_graph_leiden.cli:main",  # exposes CLI as `kgl`
        ],
    },
    python_requires=">=3.8",
)

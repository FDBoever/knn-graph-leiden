from setuptools import setup, find_packages

setup(
    name="knn-graph-leiden",
    version="0.1.0",
    description="kNN graph-based clustering with Leiden algorithm and visualisation",
    author="FDB",
    author_email="fdb@mail",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "igraph",
        "hnswlib",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "seaborn",
        "psutil"
    ],
    entry_points={
        "console_scripts": [
            "kgl=knn_graph_leiden.main_kgl:main",
            "ugl=knn_graph_leiden.main_ugl:main"
        ],
    },
    python_requires=">=3.10",
)

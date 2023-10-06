# Computational Enhancements of HNSW Targeted to Very Large Datasets
This submission to the SISAP 2023 Indexing Challenge leverages the [*hnswlib*](https://github.com/nmslib/hnswlib.git) 
implementation of the Hierarchical Navigable Small World (HNSW) index. 

> Malkov, Yu A., and Dmitry A. Yashunin. "Efficient and robust approximate nearest neighbor search using hierarchical 
> navigable small world graphs." IEEE transactions on pattern analysis and machine intelligence 42, no. 4 (2018): 
> 824-836.

To support a continuous (non-batched) construction for large datasets, this submission features a modification to the 
memory structure of the index. Unnecessary functionality is removed to optimize search-time efficiency. 

## Setup
The *hnswlib* library is written in C++ and uses PyBind11 for Python bindings. This code was tested using Python 3.8 
and Anaconda. See the [Github Workflow](https://github.com/cole-foster/sisap-2023/blob/main/.github/workflows/ci.yml)
for an example installation of this setup.

Index construction and evaluation is shown in `search/search.py`.



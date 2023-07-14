'''
    Cole Foster
    July 11th, 2023

    SISAP Indexing Challenge
'''

import argparse
import hnswlib
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time 
from utils import prepare, store_results

data_dir =  "/users/cfoste18/scratch/LAION/sisap"; # HARD-CODED
index_dir = "/users/cfoste18/scratch/LAION/sisap"; # HARD-CODED
final_dir = "/users/cfoste18/scratch/LAION/sisap"; # HARD-CODED

method = "HNSW"
def run(kind, key, size, M, ef_construction, radius1, radius2, max_neighborhood, max_neighbors):
    print(f"Running {method} on {kind}-{size}")
    
    # load dataset- download if necessary
    prepare(data_dir,kind, size)
    data = np.array(h5py.File(os.path.join(data_dir, "data", kind, size, "dataset.h5"), "r")[key],dtype=np.float32)
    queries = np.array(h5py.File(os.path.join(data_dir, "data", kind, size, "query.h5"), "r")[key],dtype=np.float32)
    n, d = data.shape
    ids = np.arange(n) # element ids

    # initialize index/data based on dataset
    if kind.startswith("pca"):
        index = hnswlib.Index(space='l2', dim=d)  # possible options are l2, cosine or ip
    elif kind.startswith("clip768"):
        print("Normalizing the Vectors")
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        index = hnswlib.Index(space='ip', dim=d) # possible options are l2, cosine or ip
    else:
        raise Exception(f"unsupported input type {kind}")

    index_identifier = f"HNSW-M-{M}-EFC-{ef_construction}"

    # load index if already created/saved
    index_save_name = os.path.join(index_dir,"index", kind, size, f"{index_identifier}.bin")
    if os.path.exists(index_save_name):
        print("Loading Index from: ",index_save_name)
        start = time.time()
        index.load_index(index_save_name, max_elements = n)
        elapsed_build = time.time() - start
        print(f"Loaded Index in {elapsed_build}s.")

    else:
        # if not, let's create and save the index
        os.makedirs(Path(index_save_name).parent, exist_ok=True)
        
        # train the index
        print(f"Training index on {data.shape}")
        start = time.time()
        index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        index.add_items(data, ids)
        elapsed_build = time.time() - start
        print(f"Done training in {elapsed_build}s.")

        # save index to file 
        index.save_index(index_save_name)
        print(f"Saved Index To: {index_save_name}")
    index.print_hierarchy()

    # search with the normal index
    # ef_vec = [10, 20, 30, 50, 70, 100, 140, 190, 250, 320, 400, 500, 650, 800, 1000]
    # for ef in ef_vec:
    #     print(f"Starting search on {queries.shape} with ef={ef}")
    #     start = time.time()
    #     index.set_ef(ef)  # ef should always be > k
    #     labels, distances = index.knn_query(queries, k=10)
    #     elapsed_search = time.time() - start
    #     print(f"Done searching in {elapsed_search}s.")
    #     labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
    #     identifier = f"index=({index_identifier}),query=(ef={ef})"
    #     store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), f"{index_identifier}", kind, distances, labels, elapsed_build, elapsed_search, identifier, size)


    # now, perform pivot selection / graph construction
    #max_neighborhood_size = 10000
    #max_neighbors = 20
    start_time = time.time()
    index.select_and_graph(radius1, radius2, max_neighborhood, max_neighbors)
    elapsed_selection = time.time() - start_time
    print(f"Built Hierarchy Index in {elapsed_selection}s.")
    index.print_hierarchy()

    # save the final index
    modified_index_identifier = f"{index_identifier}-r-{radius1:.3}-{radius2:.3}-MNH-{max_neighborhood}-MN-{max_neighbors}"
    final_index_save_name = os.path.join(final_dir,"final", kind, size, f"{modified_index_identifier}.bin")
    if not os.path.exists(final_index_save_name):
        os.makedirs(Path(final_index_save_name).parent, exist_ok=True)
    index.save_index(final_index_save_name)
    print(f"Saved New Index To: {final_index_save_name}")

    # search with the normal index
    ef_vec = [10, 20, 30, 50, 70, 100, 140, 190, 250, 320, 400, 500, 650, 800, 1000]
    for ef in ef_vec:
        print(f"Starting search on {queries.shape} with ef={ef}")
        start = time.time()
        index.set_ef(ef)  # ef should always be > k
        labels, distances = index.knn_query(queries, k=10)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({modified_index_identifier}),query=(ef={ef})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), f"{modified_index_identifier}", kind, distances, labels, elapsed_build, elapsed_search, identifier, size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        default="100K"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--EF",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "-H",
        "--max_neighborhood",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-N",
        "--max_neighbors",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    assert args.size in ["100K", "300K", "10M", "30M", "100M"]


    print("Running Modified HNSW Script With:")
    print(f"  * Dataset=clip768v2")
    print(f"  * Size={args.size}")
    print(f"  * M={args.M}                  | HNSW Parameter M")
    print(f"  * EFC={args.EF}               | HNSW Parameter ef_construction")
    print(f"  * r1={args.r1}                | radius of top layer for pivot selection")
    print(f"  * r2={args.r2}                | radius of bottom layer for pivot selection")
    print(f"  * H={args.max_neighborhood}               | maximum neighborhood size for hsp")
    print(f"  * N={args.max_neighbors}                  | max number of pivot neighbors")
    run("clip768v2", "emb", args.size, args.M, args.EF, args.r1, args.r2, args.max_neighborhood, args.max_neighbors)
